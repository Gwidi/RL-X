"""Render the fixed PD baseline, then add matched policy videos after training.

Baseline: rlx/bin/python experiments/report_landing_seed606.py
Policy:   rlx/bin/python experiments/report_landing_seed606.py --checkpoint /path/best.model
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import html
import json
import subprocess
import sys
import tempfile
import zipfile

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import (
    CONTROLLER_ID, PRESET_PATH, VIDEO_HEIGHTS, environment_config, load_preset,
    make_report_env, model_fingerprint, training_profile, verify_controller,
)
from search.main import configure_landing_collider_view, update_landing_collider_view

DEFAULT_OUT = ROOT / "docs/artifacts/seed606_training_2026-10-22"
DEFAULT_REPORT = ROOT / "docs/landing_seed606_training_2026-10-22.html"
WIDTH, HEIGHT, FPS, SPEED = 1280, 800, 40, .25
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(path, env):
    """Restore weights only; never resume optimizer or training for a video."""
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint
    from ml_collections import ConfigDict
    from rl_x.algorithms.ppo.flax.policy import get_policy
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.general_properties import GeneralProperties
    with tempfile.TemporaryDirectory(prefix="seed606_policy_") as folder:
        destination = Path(folder).resolve()
        with zipfile.ZipFile(path) as archive:
            saved_env = json.loads(archive.read("config_environment.json"))
            algorithm = json.loads(archive.read("config_algorithm.json"))
            if algorithm["name"] != "ppo.flax":
                raise ValueError("This comparison expects a ppo.flax checkpoint")
            if saved_env.get("landing_controller_id") != CONTROLLER_ID:
                raise ValueError("Checkpoint does not declare the fixed seed606 training baseline")
            if saved_env.get("landing_controller_sha256") != sha(PRESET_PATH):
                raise ValueError("Checkpoint uses a different controller preset")
            if saved_env.get("landing_model_sha256") != model_fingerprint()["combined_sha256"]:
                raise ValueError("Checkpoint and current XML/assets differ; use the original experiment model")
            if saved_env.get("imu_history_length") != env.imu_history_length:
                raise ValueError("Checkpoint IMU history does not match this experiment")
            if saved_env.get("landing_physics_sha256", "") != env.env_config.get("landing_physics_sha256", ""):
                raise ValueError("Checkpoint and evaluation use different landing physics profiles")
            from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.policy_velocity import velocity_observation_contract
            if velocity_observation_contract(saved_env.get("policy_velocity", {})) != velocity_observation_contract(env.env_config.get("policy_velocity", {})):
                raise ValueError("Checkpoint and evaluation have different policy velocity observation contracts")
            for key in ("train_robot", "spine_locked", "timestep", "control_type"):
                if saved_env.get(key) != env.env_config[key]:
                    raise ValueError(f"Checkpoint environment differs in {key}")
            for member in archive.infolist():
                if not (destination / member.filename).resolve().is_relative_to(destination):
                    raise ValueError("Invalid checkpoint archive path")
            archive.extractall(destination)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        metadata = checkpointer.metadata(str(destination))
        tree = getattr(metadata, "item_metadata", metadata)
        tree = getattr(tree, "tree", tree)
        # Read host arrays explicitly: cluster checkpoints may be sharded on a
        # GPU that does not exist on the computer rendering these videos.
        restore_args = jax.tree_util.tree_map(
            lambda _: orbax.checkpoint.RestoreArgs(restore_type=np.ndarray), tree)
        restored = checkpointer.restore(str(destination), restore_args=restore_args)
        state = restored["policy"]
        params = state.params if hasattr(state, "params") else state["params"]
    weights_hash = hashlib.sha256()
    for tree_path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        array = np.asarray(leaf)
        weights_hash.update(str(tree_path).encode())
        weights_hash.update(str(array.dtype).encode())
        weights_hash.update(str(array.shape).encode())
        weights_hash.update(array.tobytes())
    meta = SimpleNamespace(general_properties=GeneralProperties,
        single_observation_space=env.observation_space, single_action_space=env.action_space,
        policy_observation_indices=env.policy_observation_indices)
    policy, process = get_policy(SimpleNamespace(algorithm=ConfigDict(algorithm)), meta)
    @jax.jit
    def infer(observation):
        mean, _ = policy.apply(params, observation)
        return process(mean)
    def action(observation):
        value = np.asarray(jax.device_get(infer(jnp.asarray(observation)[None, :])))[0]
        if value.shape != (13,) or not np.all(np.isfinite(value)):
            raise ValueError("Invalid policy action")
        return value
    return action, dict(path=str(path.resolve()), sha256=sha(path), policy_weights_sha256=weights_hash.hexdigest(), algorithm=algorithm,
                        saved_environment=saved_env, inference="deterministic mean, clipped/rescaled as in PPO evaluation")


def render(height, controller, output, action, expected_case=None):
    wrapper = make_report_env()
    env = wrapper.env
    settings = environment_config()["landing_evaluation"]
    case = dict(settings, height=height, perturbed=False)
    wrapper.configure_landing_evaluation(case)
    observation, _ = wrapper.reset()
    identity = dict(wrapper.case_info)
    if expected_case and identity["scenario_hash"] != expected_case["scenario_hash"]:
        wrapper.close()
        raise ValueError(f"Unpaired initial state at {height} m")
    model, data = env.internal_state["mj_model"], env.internal_state["data"]
    monitor = env.internal_state["landing_safety"]
    depths = dict(foot=0., lower=0., upper=0., other=0.)
    update = monitor.update
    def measure(m, d):
        forces = update(m, d)
        mapping = monitor.contacts
        for contact in d.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if mapping.ground_geom_id not in pair:
                continue
            geom = next(g for g in pair if g != mapping.ground_geom_id)
            group = ("foot" if geom in mapping.foot_geom_ids else
                     "lower" if geom in mapping.lower_calf_geom_ids else
                     "upper" if geom in mapping.calf_geom_ids else "other")
            depths[group] = max(depths[group], -float(contact.dist) * 1000)
        return forces
    monitor.update = measure
    configure_landing_collider_view(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
    slug = f"{controller}_h_{height:g}".replace(".", "p")
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / (slug + ".mp4"))]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    next_frame = 0.
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    rewards, max_action = 0., 0.
    try:
        with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
            for _ in range(env.horizon):
                chosen = np.zeros(13) if action is None else action(observation)
                max_action = max(max_action, float(np.max(np.abs(chosen))))
                observation, reward, terminated, truncated, _ = wrapper.step(chosen)
                rewards += reward
                update_landing_collider_view(model, data)
                camera.lookat[:] = data.qpos[:3] + np.array([0., 0., -.05])
                camera.distance, camera.azimuth, camera.elevation = 1.1, 120, -12
                renderer.update_scene(data, camera=camera)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, WIDTH, 132), fill=(18, 37, 51))
                lines = ["SEED 606 · " + ("STAŁY REGULATOR PD" if controller == "pd" else "POLITYKA PPO + TEN SAM PD"),
                    f"Wysokość bazy: {height:g} m  |  t={data.time:.3f} s  |  fizyka 5 ms  |  film 4× wolniej",
                    "Zielony: stopy · Niebieski: dolna 1/3 łydki · Pomarańczowy/czerwony: górne 2/3",
                    "Stan nominalny · bez DR · " + ("zerowa akcja" if controller == "pd" else "deterministyczna akcja polityki")]
                for i, line in enumerate(lines):
                    draw.text((18, 10 + i * 29), line, fill="white", font=ImageFont.truetype(FONT, 23 if i == 0 else 19))
                done = terminated or truncated
                status = ("SUKCES" if env.internal_state["landing_success"] else
                    "PORAŻKA: " + env.internal_state["landing_failure_reason"]) if done else "W TRAKCIE"
                draw.rectangle((0, HEIGHT - 110, WIDTH, HEIGHT), fill=(18, 37, 51))
                draw.text((18, HEIGHT - 100), status, fill=(95, 230, 169) if env.internal_state["landing_success"] else "white",
                    font=ImageFont.truetype(FONT, 22))
                draw.text((18, HEIGHT - 65), f"Stopa: {monitor.peak_foot.max():.0f} N (10 ms) | Przyspieszenie: {monitor.peak_acceleration:.1f} m/s² (50 ms)",
                    fill="white", font=ImageFont.truetype(FONT, 19))
                draw.text((18, HEIGHT - 34), f"Maks. penetracja stopy: {depths['foot']:.1f} mm | Górna łydka: {monitor.peak_calf.max():.1f} N",
                    fill=(183, 210, 224), font=ImageFont.truetype(FONT, 18))
                frame = np.asarray(image).copy()
                while next_frame <= data.time:
                    process.stdin.write(frame.tobytes())
                    next_frame += SPEED / FPS
                if done:
                    break
            for _ in range(2 * FPS):
                process.stdin.write(frame.tobytes())
            image.save(output / (slug + ".png"))
        result = dict(controller=controller, height_m=height, slug=slug,
            success=bool(env.internal_state["landing_success"]),
            reason="success" if env.internal_state["landing_success"] else env.internal_state["landing_failure_reason"],
            end_time_s=float(data.time), episode_return=float(rewards), max_abs_action=max_action,
            peak_foot_force_10ms_N=float(monitor.peak_foot.max()),
            peak_upper_calf_force_10ms_N=float(monitor.peak_calf.max()),
            peak_body_acceleration_50ms_m_s2=float(monitor.peak_acceleration),
            min_base_height_m=float(monitor.min_height), max_penetration_mm=depths,
            stable_time_s=float(env.internal_state["landing_stable_time"]), case=identity)
    finally:
        process.stdin.close()
        code = process.wait()
        wrapper.close()
        if code:
            raise RuntimeError("ffmpeg failed")
    # MP4 metadata is finalized only after stdin closes and ffmpeg exits.
    result["mp4_sha256"] = sha(output / (slug + ".mp4"))
    print(controller, height, result["reason"], f"t={result['end_time_s']:.3f}", flush=True)
    return result


def build_report(output, report, pd, policy):
    def link(path):
        return os.path.relpath(path, report.parent)
    def player(row):
        return f'<video controls preload="metadata" poster="{link(output / (row["slug"] + ".png"))}"><source src="{link(output / (row["slug"] + ".mp4"))}" type="video/mp4"></video><p class="small"><strong>{"Sukces" if row["success"] else "Porażka"}</strong> · {row["end_time_s"]:.2f} s · stopa {row["peak_foot_force_10ms_N"]:.0f} N · przyspieszenie {row["peak_body_acceleration_50ms_m_s2"]:.1f} m/s² · <a href="{link(output / (row["slug"] + ".mp4"))}" download>MP4</a></p>'
    rows = []
    for row in pd["trials"]:
        learned = next((x for x in (policy or {}).get("trials", []) if x["height_m"] == row["height_m"]), None)
        right = player(learned) if learned else '<div class="empty"><strong>Polityka po treningu</strong><p>Film i metryki zostaną dodane dla dokładnie tego samego stanu początkowego i modelu.</p></div>'
        anchor = f'height_{row["height_m"]:g}'.replace(".", "p")
        rows.append(f'<section id="{anchor}"><h2>{row["height_m"]:.1f} m wysokości bazy</h2><div class="two"><div><h3>Regulator 606 · zerowa akcja</h3>{player(row)}</div><div><h3>Polityka PPO · stały PD</h3>{right}</div></div></section>')
    p = load_preset()
    status = 'Polityka: dodana z checkpointu' if policy else 'Polityka: oczekuje na trening'
    fingerprint = model_fingerprint()
    result_note = ''
    if policy:
        checkpoint = policy['checkpoint']
        result_note = (f'<section><h2>Wynik pobranego checkpointu</h2>'
            f'<p>Checkpoint: <code>{html.escape(checkpoint["path"])}</code>. '
            f'SHA256: <code>{checkpoint["sha256"]}</code>.</p>'
            '<p>Poniżej pokazano deterministyczne próby nominalne pobranego checkpointu; nie zastępują one '
            'ewaluacji statystycznej z zaburzeniami.</p></section>')
    text = f'''<!doctype html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Seed606 — stały PD i przyszłe porównanie PPO</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f7;color:#193346;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1450px;margin:auto;padding:30px}}h1{{font-size:40px;line-height:1.18;max-width:1000px}}h2{{font-size:28px;line-height:1.3}}h3{{font-size:18px}}section{{background:white;border:1px solid #dbe5ec;padding:28px;border-radius:10px;margin:24px 0}}a{{color:#226e9d}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}video{{width:100%;height:auto;border-radius:8px;background:#102a3b}}.small{{font-size:13px;color:#607789}}.empty{{min-height:280px;border:2px dashed #cad8e2;background:#f7fafb;border-radius:8px;display:flex;flex-direction:column;justify-content:center;padding:30px;color:#607789;text-align:center}}.callout{{background:#edf4f8;border-left:3px solid #347e9f;padding:18px 22px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dbe5ec}}th{{background:#edf4f8}}pre{{overflow:auto;background:#edf3f7;padding:18px;border-radius:6px;font-size:13px}}code{{overflow-wrap:anywhere}}.status{{font-size:13px;letter-spacing:.04em;color:#347965;font-weight:700}}.scroll{{overflow-x:auto}}@media(max-width:900px){{.two{{grid-template-columns:1fr}}main{{padding:16px}}section{{padding:18px}}h1{{font-size:31px}}}}@media print{{video{{display:none}}.two{{grid-template-columns:1fr 1fr}}section{{break-inside:avoid}}}}
</style></head><body><main><p><a href="{link(ROOT/'docs/landing_status_2026-10-22.html')}">Historia analizy PPO i search</a></p><div class="status">NOWY EKSPERYMENT · STAŁE KP/KD · {status.upper()}</div><h1>Regulator 606 jako baza treningu PPO</h1>
<p>Jedne nastawy i jedna pozycja bazowa na czterech wysokościach. Filmy powstały w rzeczywistym środowisku PPO, z zerową akcją, nominalną fizyką i kryterium stabilizacji. Wysokości oznaczają położenie bazy nad płaskim podłożem, bez dodawania wysokości home.</p>
<p>Przejdź do filmów: <a href="#height_2p5">2,5 m</a> · <a href="#height_2">2,0 m</a> · <a href="#height_1p5">1,5 m</a> · <a href="#height_1">1,0 m</a>.</p>
<section><h2>Konfiguracja uczenia</h2><div class="scroll"><table><tr><th>Element</th><th>Ustawienie</th></tr><tr><td>Baza PD</td><td>Seed 606, optimum z 2,5 m; wartości z zaokrąglonego stdout klastra.</td></tr><tr><td>Kp nóg / Kd nóg</td><td>{p['gains']['kp_hip']} / {p['gains']['kp_thigh']} / {p['gains']['kp_calf']} oraz {p['gains']['kd_hip']} / {p['gains']['kd_thigh']} / {p['gains']['kd_calf']} (biodro / udo / łydka).</td></tr><tr><td>Spine</td><td>Aktywny; Kp={p['gains']['kp_spine']}, Kd={p['gains']['kd_spine']}.</td></tr><tr><td>Akcja</td><td>13 odchyleń pozycji wokół home, skala 0,6 rad, ograniczenia stawów; bez wyboru Kp/Kd przez politykę.</td></tr><tr><td>Curriculum</td><td>Baza 1,0 → 2,5 m; przechył 0 → ±3°; początkowa prędkość XY 0 → ±3 m/s na oś; 15% łatwego replay.</td></tr><tr><td>Fizyka i kontakt</td><td>5 ms, sterowanie 50 Hz; górne 2/3 łydki zabronione, dolna 1/3 dozwolona.</td></tr><tr><td>Randomizacja — faza 1</td><td>Współczynnik DR=0; parametry modelu, PD, home i czujników nominalne. Curriculum stanu początkowego pozostaje aktywne.</td></tr><tr><td>PPO</td><td>Flax; 64 środowiska × 2048 kroków; lr=3e-5; std=0,15; 10 epok; minibatch 64; budżet 30 015 488 przejść.</td></tr><tr><td>Ewaluacja</td><td>Co 1 048 576 przejść: polityka i zero-action PD, nominalnie i z przechyłem/prędkością, na 1,0 / 1,5 / 2,0 / 2,5 m. Nowy seed 20261023; 64 epizody na wysokość, warunek i kontroler.</td></tr></table></div>
<p>Wybór Kd raz na epizod jest odłożony. Rozpoczynamy nowy trening ze stałą bazą, bez ładowania poprzednich wag. Zgodność XML i meshy jest sprawdzana przed startem i zapisana w konfiguracji checkpointu. Szeroka randomizacja fizyki to późniejszy, osobno oznaczony eksperyment.</p>
<pre>sbatch experiments/test2.sh</pre><p class="small">Uruchomienie z katalogu głównego projektu albo <code>sbatch test2.sh</code> z katalogu experiments. test2.sh uruchamia wspólny profil seeda 606. Przygotowano i sprawdzono lokalną próbą 32 przejść PPO; pełnego zadania klastrowego nie uruchomiono. To kontrola działania ścieżki uczenia, nie wynik wytrenowanej polityki.</p></section>
{''.join(rows)}
<section><h2>Warunki porównania po treningu</h2><p>Regulator i polityka dostają ten sam model, home, Kp/Kd, stan początkowy, historię IMU przy resecie, wysokości oraz limit 5 s. Polityka działa deterministycznie, ze średnią rozkładu akcji i tym samym clippingiem co w ewaluacji PPO. Generator odrzuca niezgodny model, preset lub początkowy hash scenariusza. Filmy pokazują pojedyncze próby nominalne; skuteczność statystyczną oceniamy osobną sparowaną ewaluacją.</p>
<pre>rlx/bin/python experiments/report_landing_seed606.py --checkpoint /sciezka/do/best.model</pre>
<p>Domyślny raport zachowa zapisane filmy PD i doda politykę do prawej kolumny. Odtwarzanie samej bazy: <code>rlx/bin/python experiments/report_landing_seed606.py</code>.</p>
<div class="callout"><strong>Interpretacja fizyczna:</strong> parametry miękkiego kontaktu stóp pozostały takie same. Raport zachowuje pomiary penetracji w JSON; wysokiej skuteczności w tej symulacji nie należy utożsamiać z kalibracją kontaktu na sprzęcie. Krok czasu i model kontaktu muszą pozostać wspólne w porównaniu PD–PPO.</div>
<p class="small">Dane: <a href="{link(output/'pd_results.json')}">wyniki regulatora</a> · <a href="{link(output/'protocol.json')}">zamrożony protokół i hashe</a> · <a href="{link(output/'training_profile.json')}">profil uruchomienia</a> · <a href="{link(output/'environment.json')}">pełna konfiguracja środowiska</a> · <a href="{link(PRESET_PATH)}">preset PD/home</a>. Model: <code>{fingerprint['combined_sha256']}</code>.</p></section></main></body></html>'''
    text = text.replace('<section><h2>Konfiguracja uczenia</h2>', result_note + '<section><h2>Konfiguracja uczenia</h2>')
    if policy:
        text = text.replace('Przygotowano i sprawdzono lokalną próbą 32 przejść PPO; pełnego zadania klastrowego nie uruchomiono. To kontrola działania ścieżki uczenia, nie wynik wytrenowanej polityki.', 'Trening klastrowy został zakończony; filmy PPO pochodzą z checkpointu wskazanego powyżej.')
    report.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    fingerprint = verify_controller()
    core_files = [
        "rl_x/environments/custom_mujoco/robot_locomotion/mujoco/" + name for name in (
            "environment.py", "default_config.py", "landing_safety.py", "landing_evaluation.py",
            "control_functions/pd.py", "reward_functions/simplified_landing_reward.py",
            "domain_randomization/initial_state_functions/energy_dissipation_curriculum.py")]
    core_hashes = {name: sha(ROOT / name) for name in core_files}
    action = metadata = None
    if args.checkpoint:
        wrapper = make_report_env()
        try:
            action, metadata = load_policy(args.checkpoint, wrapper.env)
            wrapper.configure_landing_evaluation(dict(environment_config()["landing_evaluation"], height=1., perturbed=False))
            observation, _ = wrapper.reset()
            value = action(observation)
            print("Checkpoint inference verified:", value.shape, "max abs", float(np.max(np.abs(value))), flush=True)
        finally:
            wrapper.close()
    if args.check_only:
        print(json.dumps(fingerprint, indent=2))
        return
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["model"]["combined_sha256"] != fingerprint["combined_sha256"]:
            raise ValueError("The saved comparison uses a different model; preserve it and choose a new output directory")
        if protocol.get("environment_core_sha256", core_hashes) != core_hashes:
            raise ValueError("Environment code changed since the PD videos; preserve this experiment and use a new output directory")
    else:
        protocol = dict(version=1, controller_id=CONTROLLER_ID, controller_sha256=sha(PRESET_PATH),
            model=fingerprint, heights_base_m=list(VIDEO_HEIGHTS), seed=20261023,
            nominal=True, observation_history=5, deterministic_policy=True, physics_dt=.005,
            control_dt=.02, episode_limit_s=5, video=dict(width=WIDTH, height=HEIGHT, fps=FPS, speed=SPEED),
            case_settings=environment_config()["landing_evaluation"])
        protocol_path.write_text(json.dumps(protocol, indent=2))
        (output / "training_profile.json").write_text(json.dumps(training_profile("seed606_fixed_pd"), indent=2))
        (output / "environment.json").write_text(json.dumps(environment_config(), indent=2))
    protocol["environment_core_sha256"] = core_hashes
    protocol_path.write_text(json.dumps(protocol, indent=2))
    pd_path = output / "pd_results.json"
    if pd_path.exists():
        pd = json.loads(pd_path.read_text())
    else:
        pd = dict(controller="fixed_pd_zero_action", trials=[render(h, "pd", output, None) for h in VIDEO_HEIGHTS])
        pd_path.write_text(json.dumps(pd, indent=2))
    policy_path = output / "policy_results.json"
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else None
    if action:
        policy = dict(checkpoint=metadata, trials=[render(row["height_m"], "policy", output, action, row["case"]) for row in pd["trials"]])
        policy_path.write_text(json.dumps(policy, indent=2))
    build_report(output, args.report, pd, policy)
    print("Report:", args.report, flush=True)


if __name__ == "__main__":
    main()
