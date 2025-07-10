import os
import json
import gradio as gr
from PIL import Image
from uuid import uuid4
from typing import List
from collections import defaultdict
from datetime import datetime, timedelta

from modules import call_queue, shared, script_callbacks, scripts, ui_components
from modules.shared import list_checkpoint_tiles, refresh_checkpoints
from modules.cmd_args import parser
from modules.ui import create_refresh_button
from modules.ui_common import save_files
from modules.sd_models import model_path
from modules.generation_parameters_copypaste import (
    registered_param_bindings,
    register_paste_params_button,
    connect_paste_params_buttons,
    parse_generation_parameters,
    ParamBinding,
)

from agent_scheduler.task_runner import TaskRunner, get_instance
from agent_scheduler.helpers import log, compare_components_with_ids, get_components_by_ids, is_macos, i18n
from agent_scheduler.db import init as init_db, task_manager, TaskStatus
from agent_scheduler.api import regsiter_apis

is_sdnext = parser.description == "SD.Next"
try:
    import ldm_patched.modules.model_management
    is_forge = True
except ImportError:
    is_forge = False

ToolButton = gr.Button if is_sdnext or is_forge else ui_components.ToolButton

task_runner: TaskRunner = None

checkpoint_current = "Current Checkpoint"
checkpoint_runtime = "Runtime Checkpoint"
queue_with_every_checkpoints = "$$_queue_with_all_checkpoints_$$"

ui_placement_as_tab = "As a tab"
ui_placement_append_to_main = "Append to main UI"

placement_under_generate = "Under Generate button"
placement_between_prompt_and_generate = "Between Prompt and Generate button"

completion_action_choices = ["Do nothing", "Shut down", "Restart", "Sleep", "Hibernate", "Stop webui"]

task_filter_choices = ["All", "Bookmarked", "Done", "Failed", "Interrupted"]

enqueue_key_modifiers = [
    "Command" if is_macos else "Ctrl",
    "Control" if is_macos else "Alt",
    "Shift",
]
enqueue_default_hotkey = enqueue_key_modifiers[0] + "+KeyE"
enqueue_key_codes = {}
enqueue_key_codes.update({chr(i): "Key" + chr(i) for i in range(ord("A"), ord("Z") + 1)})
enqueue_key_codes.update({chr(i): "Digit" + chr(i) for i in range(ord("0"), ord("9") + 1)})
enqueue_key_codes.update({"`": "Backquote", "Enter": "Enter"})

task_history_retenion_map = {
    "1 day": 1,
    "3 days": 3,
    "7 days": 7,
    "14 days": 14,
    "30 days": 30,
    "90 days": 90,
    "Keep forever": 0,
}

init_db()


class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        script_callbacks.on_app_started(lambda block, _: self.on_app_started(block))
        self.checkpoint_override = checkpoint_current
        self.generate_button = None
        self.gallery = None
        self.enqueue_row = None
        self.checkpoint_dropdown = None
        self.submit_button = None

    def title(self):
        return i18n("Agent Scheduler")

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def on_checkpoint_changed(self, checkpoint):
        self.checkpoint_override = checkpoint

    def after_component(self, component, **_kwargs):
        generate_id = "txt2img_generate" if self.is_txt2img else "img2img_generate"
        gallery_id = "txt2img_gallery" if self.is_txt2img else "img2img_gallery"
        generate_box = "txt2img_generate_box" if self.is_txt2img else "img2img_generate_box"
        actions_column_id = "txt2img_actions_column" if self.is_txt2img else "img2img_actions_column"
        neg_id = "txt2img_neg_prompt" if self.is_txt2img else "img2img_neg_prompt"
        toprow_id = "txt2img_toprow" if self.is_txt2img else "img2img_toprow"

        if component.elem_id == gallery_id:
            self.gallery = component

        def add_enqueue_row(elem_id):
            parent = component.parent
            while parent is not None:
                if parent.elem_id == elem_id:
                    self.add_enqueue_button()
                    component.parent.children.pop()
                    parent.add(self.enqueue_row)
                    break
                parent = parent.parent

        if component.elem_id == generate_id:
            self.generate_button = component
            if getattr(shared.opts, "compact_prompt_box", False):
                add_enqueue_row(generate_box)
            else:
                if getattr(shared.opts, "queue_button_placement", placement_under_generate) == placement_under_generate:
                    add_enqueue_row(actions_column_id)
        elif component.elem_id == neg_id:
            if not getattr(shared.opts, "compact_prompt_box", False):
                if getattr(shared.opts, "queue_button_placement", placement_under_generate) == placement_between_prompt_and_generate:
                    add_enqueue_row(toprow_id)

    def on_app_started(self, block):
        if self.generate_button is not None:
            self.bind_enqueue_button(block)

    def add_enqueue_button(self):
        id_part = "img2img" if self.is_img2img else "txt2img"
        with gr.Row(elem_id=f"{id_part}_enqueue_wrapper") as row:
            self.enqueue_row = row
            hide_checkpoint = getattr(shared.opts, "queue_button_hide_checkpoint", True)
            self.checkpoint_dropdown = gr.Dropdown(
                choices=get_checkpoint_choices(),
                value=checkpoint_current,
                show_label=False,
                interactive=True,
                visible=not hide_checkpoint,
            )
            if not hide_checkpoint:
                create_refresh_button(
                    self.checkpoint_dropdown,
                    refresh_checkpoints,
                    lambda: {"choices": get_checkpoint_choices()},
                    f"refresh_{id_part}_checkpoint",
                )
            self.submit_button = gr.Button(i18n("Enqueue"), elem_id=f"{id_part}_enqueue", variant="primary")

    def bind_enqueue_button(self, root):
        is_img2img = self.is_img2img

        if self.gallery is None:
            log.error("[AgentScheduler] Could not find the main gallery component. The 'Enqueue' button will not work.")
            return

        # Find the main dependency for the generate button
        dependency = None
        api_name = "img2img" if self.is_img2img else "txt2img"
        
        for d in root.config['dependencies']:
            if d.get('api_name') == api_name:
                dependency = d
                break
        
        if dependency is None:
            # Fallback to old method if api_name is not found
            log.warning(f"[AgentScheduler] Could not find api_name:{api_name}, fallback to old method.")
            # Find dependencies that are triggered by the generate button and output to the gallery
            click_deps = [
                d for d in root.config['dependencies']
                if d.get('trigger') == 'click' and self.generate_button._id in d.get('targets', [])
            ]
            
            gallery_deps = [d for d in click_deps if self.gallery._id in d.get('outputs', [])]

            if len(gallery_deps) == 1:
                dependency = gallery_deps[0]
            elif gallery_deps:
                # If multiple, pick the one with the most inputs
                dependency = max(gallery_deps, key=lambda d: len(d.get('inputs', [])))
            elif click_deps:
                # Fallback if no dependency outputs to the gallery (e.g. API only)
                # Pick the one with the most inputs from all click dependencies
                dependency = max(click_deps, key=lambda d: len(d.get('inputs', [])))

        if dependency is None:
            log.error("[AgentScheduler] Could not find the main generation dependency. The 'Enqueue' button will not work.")
            return

        with root:
            if self.checkpoint_dropdown is not None:
                self.checkpoint_dropdown.change(fn=self.on_checkpoint_changed, inputs=[self.checkpoint_dropdown])

            fn = self.wrap_register_ui_task()

            # Resolve component IDs to component objects
            raw_inputs = dependency.get("inputs", [])
            log.debug(f"[AgentScheduler] Found dependency with raw_inputs: {raw_inputs}")

            input_components = get_components_by_ids(root, raw_inputs)

            log.debug(f"[AgentScheduler] Resolved {len(input_components)} components out of {len(raw_inputs)} raw inputs.")

            if len(input_components) != len(raw_inputs):
                log.warning(f"[AgentScheduler] Could not resolve all components. Some components may be missing from the task.")
            
            # Prepend the checkpoint dropdown if it exists
            if self.checkpoint_dropdown:
                input_components.insert(0, self.checkpoint_dropdown)

            args = dict(
                fn=fn,
                _js="submit_enqueue_img2img" if is_img2img else "submit_enqueue",
                inputs=input_components,
                outputs=None,
                show_progress=False,
            )

            self.submit_button.click(**args)

    def wrap_register_ui_task(self):
        def f(request: gr.Request, *args):
            if len(args) < 2:
                raise Exception("Invalid call")

            task_id = args[0]
            checkpoint: str = args[1]
            args = args[2:]
            log.info(f"[{self.title()}] Registering task '{task_id}' with checkpoint '{checkpoint}'")
            task_name = None

            if task_id == queue_with_every_checkpoints:
                task_id = str(uuid4())
                checkpoint = list_checkpoint_tiles()
            else:
                if not task_id.startswith("task("):
                    task_name = task_id
                    task_id = str(uuid4())

                if checkpoint is None or checkpoint == "" or checkpoint == checkpoint_current:
                    checkpoint = [shared.sd_model.sd_checkpoint_info.title]
                elif checkpoint == checkpoint_runtime:
                    checkpoint = [None]
                elif checkpoint.endswith(" checkpoints)"):
                    checkpoint_dir = " ".join(checkpoint.split(" ")[0:-2])
                    checkpoint = list(filter(lambda c: c.startswith(checkpoint_dir), list_checkpoint_tiles()))
                else:
                    checkpoint = [checkpoint]

            log.info(f"[{self.title()}] Queuing {len(checkpoint)} tasks.")
            for i, c in enumerate(checkpoint):
                t_id = task_id if i == 0 else f"{task_id}.{i}"
                task_runner.register_ui_task(
                    t_id,
                    self.is_img2img,
                    *args,
                    checkpoint=c,
                    task_name=task_name,
                    request=request,
                )

            task_runner.execute_pending_tasks_threading()

        return f


def get_checkpoint_choices():
    checkpoints: List[str] = list_checkpoint_tiles()

    checkpoint_dirs = defaultdict(lambda: 0)
    for checkpoint in checkpoints:
        checkpoint_dir = os.path.dirname(checkpoint)
        while checkpoint_dir != "" and checkpoint_dir != "/":
            checkpoint_dirs[checkpoint_dir] += 1
            checkpoint_dir = os.path.dirname(checkpoint_dir)

    choices = checkpoints
    choices.extend([f"{d} ({checkpoint_dirs[d]} checkpoints)" for d in checkpoint_dirs.keys()])
    choices = sorted(choices)

    choices.insert(0, checkpoint_runtime)
    choices.insert(0, checkpoint_current)

    return choices


def create_send_to_buttons():
    return {
        "txt2img": ToolButton(
            "➠ text" if is_sdnext or is_forge else "📝",
            elem_id="agent_scheduler_send_to_txt2img",
            tooltip=i18n("Send generation parameters to txt2img tab."),
        ),
        "img2img": ToolButton(
            "➠ image" if is_sdnext or is_forge else "🖼️",
            elem_id="agent_scheduler_send_to_img2img",
            tooltip=i18n("Send image and generation parameters to img2img tab."),
        ),
        "inpaint": ToolButton(
            "➠ inpaint" if is_sdnext or is_forge else "🎨️",
            elem_id="agent_scheduler_send_to_inpaint",
            tooltip=i18n("Send image and generation parameters to img2img inpaint tab."),
        ),
        "extras": ToolButton(
            "➠ process" if is_sdnext or is_forge else "📐",
            elem_id="agent_scheduler_send_to_extras",
            tooltip=i18n("Send image and generation parameters to extras tab."),
        ),
    }


def infotexts_to_geninfo(infotexts: List[str]):
    all_promts = []
    all_seeds = []

    geninfo = {"infotexts": infotexts, "all_prompts": all_promts, "all_seeds": all_seeds, "index_of_first_image": 0}

    for infotext in infotexts:
        # Dynamic prompt breaks layout of infotext
        if "Template: " in infotext:
            lines = infotext.split("\n")
            lines = [l for l in lines if not (l.startswith("Template: ") or l.startswith("Negative Template: "))]
            infotext = "\n".join(lines)

        params = parse_generation_parameters(infotext)

        if "prompt" not in geninfo:
            geninfo["prompt"] = params.get("Prompt", "")
            geninfo["negative_prompt"] = params.get("Negative prompt", "")
            geninfo["seed"] = params.get("Seed", "-1")
            geninfo["sampler_name"] = params.get("Sampler", "")
            geninfo["cfg_scale"] = params.get("CFG scale", "")
            geninfo["steps"] = params.get("Steps", "0")
            geninfo["width"] = params.get("Size-1", "512")
            geninfo["height"] = params.get("Size-2", "512")

        all_promts.append(params.get("Prompt", ""))
        all_seeds.append(params.get("Seed", "-1"))

    return geninfo


def get_task_results(task_id: str, image_idx: int = None):
    task = task_manager.get_task(task_id)

    galerry = None
    geninfo = None
    infotext = None
    if task is None:
        pass
    elif task.status != TaskStatus.DONE:
        infotext = f'{i18n("Status: ")}{task.status}'
        if task.status == TaskStatus.FAILED and task.result:
            infotext += f'\n{i18n("Error: ")}{task.result}'
    elif task.status == TaskStatus.DONE:
        try:
            result: dict = json.loads(task.result)
            images = result.get("images", [])
            geninfo = result.get("geninfo", None)
            if isinstance(geninfo, dict):
                infotexts = geninfo.get("infotexts", [])
            else:
                infotexts = result.get("infotexts", [])
                geninfo = infotexts_to_geninfo(infotexts)

            galerry = [Image.open(i) for i in images if os.path.exists(i)] if image_idx is None else gr.update()
            idx = image_idx if image_idx is not None else 0
            if idx < len(infotexts):
                infotext = infotexts[idx]
        except Exception as e:
            log.error(f"[AgentScheduler] Failed to load task result")
            log.error(e)
            infotext = f'{i18n("Failed to load task result: ")}{str(e)}'

    res = (
        gr.Textbox.update(infotext, visible=infotext is not None),
        gr.Row.update(visible=galerry is not None),
    )

    if image_idx is None:
        geninfo = json.dumps(geninfo) if geninfo else None
        res += (
            galerry,
            gr.Textbox.update(geninfo),
            gr.File.update(None, visible=False),
            gr.HTML.update(None),
        )

    return res


def remove_old_tasks():
    # delete task that are too old

    retention_days = 30
    if (
        getattr(shared.opts, "queue_history_retention_days", None)
        and shared.opts.queue_history_retention_days in task_history_retenion_map
    ):
        retention_days = task_history_retenion_map[shared.opts.queue_history_retention_days]

    if retention_days > 0:
        deleted_rows = task_manager.delete_tasks(before=datetime.now() - timedelta(days=retention_days))
        if deleted_rows > 0:
            log.debug(f"[AgentScheduler] Deleted {deleted_rows} tasks older than {retention_days} days")


def on_ui_tab(**_kwargs):
    grid_page_size = getattr(shared.opts, "queue_grid_page_size", 0)

    with gr.Blocks(analytics_enabled=False) as scheduler_tab:
        with gr.Tabs(elem_id="agent_scheduler_tabs"):
            with gr.Tab(i18n("Task Queue"), id=0, elem_id="agent_scheduler_pending_tasks_tab"):
                with gr.Row(elem_id="agent_scheduler_pending_tasks_wrapper"):
                    with gr.Column(scale=1):
                        with gr.Row(elem_id="agent_scheduler_pending_tasks_actions", elem_classes="flex-row"):
                            paused = getattr(shared.opts, "queue_paused", False)

                            gr.Button(
                                i18n("Pause"),
                                elem_id="agent_scheduler_action_pause",
                                variant="stop",
                                visible=not paused,
                            )
                            gr.Button(
                                i18n("Resume"),
                                elem_id="agent_scheduler_action_resume",
                                variant="primary",
                                visible=paused,
                            )
                            gr.Button(
                                i18n("Refresh"),
                                elem_id="agent_scheduler_action_reload",
                                variant="secondary",
                            )
                            gr.Button(
                                i18n("Clear"),
                                elem_id="agent_scheduler_action_clear_queue",
                                variant="stop",
                            )
                            gr.Button(
                                i18n("Export"),
                                elem_id="agent_scheduler_action_export",
                                variant="secondary",
                            )
                            gr.Button(
                                i18n("Import"),
                                elem_id="agent_scheduler_action_import",
                                variant="secondary",
                            )
                            gr.HTML(f'<input type="file" id="agent_scheduler_import_file" style="display: none" accept="application/json" />')

                            with gr.Row(elem_classes=["agent_scheduler_filter_container", "flex-row", "ml-auto"]):
                                gr.Textbox(
                                    max_lines=1,
                                    placeholder=i18n("Search"),
                                    label=i18n("Search"),
                                    show_label=False,
                                    min_width=0,
                                    elem_id="agent_scheduler_action_search",
                                )
                        gr.HTML(
                            f'<div id="agent_scheduler_pending_tasks_grid" class="ag-theme-gradio" data-page-size="{grid_page_size}"></div>'
                        )
                    with gr.Column(scale=1):
                        gr.Gallery(
                            elem_id="agent_scheduler_current_task_images",
                            label=i18n("Output"),
                            show_label=False,
                            columns=2,
                            object_fit="contain",
                        )
            with gr.Tab(i18n("Task History"), id=1, elem_id="agent_scheduler_history_tab"):
                with gr.Row(elem_id="agent_scheduler_history_wrapper"):
                    with gr.Column(scale=1):
                        with gr.Row(elem_id="agent_scheduler_history_actions", elem_classes="flex-row"):
                            gr.Button(
                                i18n("Requeue Failed"),
                                elem_id="agent_scheduler_action_requeue",
                                variant="primary",
                            )
                            gr.Button(
                                i18n("Refresh"),
                                elem_id="agent_scheduler_action_refresh_history",
                                elem_classes="agent_scheduler_action_refresh",
                                variant="secondary",
                            )
                            gr.Button(
                                i18n("Clear"),
                                elem_id="agent_scheduler_action_clear_history",
                                variant="stop",
                            )

                            with gr.Row(elem_classes=["agent_scheduler_filter_container", "flex-row", "ml-auto"]):
                                status = gr.Dropdown(
                                    elem_id="agent_scheduler_status_filter",
                                    choices=[i18n(c) for c in task_filter_choices],
                                    value=i18n("All"),
                                    show_label=False,
                                    min_width=0,
                                )
                                gr.Textbox(
                                    max_lines=1,
                                    placeholder=i18n("Search"),
                                    label=i18n("Search"),
                                    show_label=False,
                                    min_width=0,
                                    elem_id="agent_scheduler_action_search_history",
                                )
                        gr.HTML(
                            f'<div id="agent_scheduler_history_tasks_grid" class="ag-theme-gradio" data-page-size="{grid_page_size}"></div>'
                        )
                    with gr.Column(scale=1, elem_id="agent_scheduler_history_results"):
                        galerry = gr.Gallery(
                            elem_id="agent_scheduler_history_gallery",
                            label=i18n("Output"),
                            show_label=False,
                            columns=2,
                            preview=True,
                            object_fit="contain",
                        )
                        with gr.Row(
                            elem_id="agent_scheduler_history_result_actions",
                            visible=False,
                        ) as result_actions:
                            if is_sdnext or is_forge:
                                with gr.Group():
                                    save = ToolButton(
                                        "💾",
                                        elem_id="agent_scheduler_save",
                                        tooltip=i18n(f"Save the image to a dedicated directory ({shared.opts.outdir_save})."),
                                    )
                                    save_zip = None
                            else:
                                save = ToolButton(
                                    "💾",
                                    elem_id="agent_scheduler_save",
                                    tooltip=i18n(f"Save the image to a dedicated directory ({shared.opts.outdir_save})."),
                                )
                                save_zip = ToolButton(
                                    "🗃️",
                                    elem_id="agent_scheduler_save_zip",
                                    tooltip=i18n(f"Save zip archive with images to a dedicated directory ({shared.opts.outdir_save})"),
                                )
                            send_to_buttons = create_send_to_buttons()
                        with gr.Group():
                            generation_info = gr.Textbox(visible=False, elem_id=f"agent_scheduler_generation_info")
                            infotext = gr.TextArea(
                                label=i18n("Generation Info"),
                                elem_id=f"agent_scheduler_history_infotext",
                                interactive=False,
                                visible=True,
                                lines=3,
                            )
                            download_files = gr.File(
                                None,
                                file_count="multiple",
                                interactive=False,
                                show_label=False,
                                visible=False,
                                elem_id=f"agent_scheduler_download_files",
                            )
                            html_log = gr.HTML(elem_id=f"agent_scheduler_html_log", elem_classes="html-log")
                            selected_task = gr.Textbox(
                                elem_id="agent_scheduler_history_selected_task",
                                visible=False,
                                show_label=False,
                            )
                            selected_image_id = gr.Textbox(
                                elem_id="agent_scheduler_history_selected_image",
                                visible=False,
                                show_label=False,
                            )

        # register event handlers
        status.change(
            fn=lambda x: None,
            _js="agent_scheduler_status_filter_changed",
            inputs=[status],
        )
        save.click(
            fn=lambda x, y, z: call_queue.wrap_gradio_call(save_files)(x, y, False, int(z)),
            _js="(x, y, z) => [x, y, selected_gallery_index()]",
            inputs=[generation_info, galerry, infotext],
            outputs=[download_files, html_log],
            show_progress=False,
        )
        if save_zip:
            save_zip.click(
                fn=lambda x, y, z: call_queue.wrap_gradio_call(save_files)(x, y, True, int(z)),
                _js="(x, y, z) => [x, y, selected_gallery_index()]",
                inputs=[generation_info, galerry, infotext],
                outputs=[download_files, html_log],
            )
        selected_task.change(
            fn=lambda x: get_task_results(x, None),
            inputs=[selected_task],
            outputs=[infotext, result_actions, galerry, generation_info, download_files, html_log],
        )
        selected_image_id.change(
            fn=lambda x, y: get_task_results(x, image_idx=int(y)),
            inputs=[selected_task, selected_image_id],
            outputs=[infotext, result_actions],
        )
        try:
            for paste_tabname, paste_button in send_to_buttons.items():
                register_paste_params_button(
                    ParamBinding(
                        paste_button=paste_button,
                        tabname=paste_tabname,
                        source_text_component=infotext,
                        source_image_component=galerry,
                    )
                )
        except:
            pass

    return [(scheduler_tab, "Agent Scheduler", "agent_scheduler")]


def on_ui_settings():
    section = ("agent_scheduler", i18n("Agent Scheduler"))
    shared.opts.add_option(
        "agent_scheduler_language",
        shared.OptionInfo(
            "English",
            "Language",
            gr.Radio,
            lambda: {
                "choices": ["English", "中文"],
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_paused",
        shared.OptionInfo(
            False,
            i18n("Disable queue auto-processing"),
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_button_hide_checkpoint",
        shared.OptionInfo(
            True,
            i18n("Hide the custom checkpoint dropdown"),
            gr.Checkbox,
            {},
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_button_placement",
        shared.OptionInfo(
            placement_under_generate,
            i18n("Queue button placement"),
            gr.Radio,
            lambda: {
                "choices": [
                    i18n(placement_under_generate),
                    i18n(placement_between_prompt_and_generate),
                ]
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_ui_placement",
        shared.OptionInfo(
            ui_placement_as_tab,
            i18n("Task queue UI placement"),
            gr.Radio,
            lambda: {
                "choices": [
                    i18n(ui_placement_as_tab),
                    i18n(ui_placement_append_to_main),
                ]
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_history_retention_days",
        shared.OptionInfo(
            "30 days",
            i18n("Auto delete queue history (bookmarked tasks excluded)"),
            gr.Radio,
            lambda: {
                "choices": [i18n(c) for c in task_history_retenion_map.keys()],
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_automatic_requeue_failed_task",
        shared.OptionInfo(
            False,
            i18n("Auto requeue failed tasks"),
            gr.Checkbox,
            {},
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_grid_page_size",
        shared.OptionInfo(
            0,
            i18n("Task list page size (0 for auto)"),
            gr.Slider,
            {"minimum": 0, "maximum": 200, "step": 1},
            section=section,
        ),
    )

    def enqueue_keyboard_shortcut(disabled: bool, modifiers, key_code: str):
        if disabled:
            modifiers.insert(0, "Disabled")

        shortcut = "+".join(sorted(modifiers) + [enqueue_key_codes[key_code]])

        return (
            shortcut,
            gr.CheckboxGroup.update(interactive=not disabled),
            gr.Dropdown.update(interactive=not disabled),
        )

    def enqueue_keyboard_shortcut_ui(**_kwargs):
        value = _kwargs.get("value", enqueue_default_hotkey)
        parts = value.split("+")
        key = parts.pop()
        key_code_value = [k for k, v in enqueue_key_codes.items() if v == key]
        modifiers = [m for m in parts if m in enqueue_key_modifiers]
        disabled = "Disabled" in value

        with gr.Group(elem_id="enqueue_keyboard_shortcut_wrapper"):
            modifiers = gr.CheckboxGroup(
                enqueue_key_modifiers,
                value=modifiers,
                label=i18n("Enqueue keyboard shortcut"),
                elem_id="enqueue_keyboard_shortcut_modifiers",
                interactive=not disabled,
            )
            key_code = gr.Dropdown(
                choices=list(enqueue_key_codes.keys()),
                value="E" if len(key_code_value) == 0 else key_code_value[0],
                elem_id="enqueue_keyboard_shortcut_key",
                label=i18n("Key"),
                interactive=not disabled,
            )
            shortcut = gr.Textbox(**_kwargs)
            disable = gr.Checkbox(
                value=disabled,
                elem_id="enqueue_keyboard_shortcut_disable",
                label=i18n("Disable keyboard shortcut"),
            )

        modifiers.change(
            fn=enqueue_keyboard_shortcut,
            inputs=[disable, modifiers, key_code],
            outputs=[shortcut, modifiers, key_code],
        )
        key_code.change(
            fn=enqueue_keyboard_shortcut,
            inputs=[disable, modifiers, key_code],
            outputs=[shortcut, modifiers, key_code],
        )
        disable.change(
            fn=enqueue_keyboard_shortcut,
            inputs=[disable, modifiers, key_code],
            outputs=[shortcut, modifiers, key_code],
        )

        return shortcut

    shared.opts.add_option(
        "queue_keyboard_shortcut",
        shared.OptionInfo(
            enqueue_default_hotkey,
            i18n("Enqueue keyboard shortcut"),
            enqueue_keyboard_shortcut_ui,
            {
                "interactive": False,
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "queue_completion_action",
        shared.OptionInfo(
            "Do nothing",
            i18n("Action after queue completion"),
            gr.Radio,
            lambda: {
                "choices": [i18n(c) for c in completion_action_choices],
            },
            section=section,
        ),
    )


def on_app_started(block, app):
    global task_runner
    task_runner = get_instance(block)
    task_runner.execute_pending_tasks_threading()
    regsiter_apis(app, task_runner)
    task_runner.on_task_cleared(lambda: remove_old_tasks())

    if getattr(shared.opts, "queue_ui_placement", "") == ui_placement_append_to_main and block:
        with block:
            with block.children[1]:
                bindings = registered_param_bindings.copy()
                registered_param_bindings.clear()
                on_ui_tab()
                connect_paste_params_buttons()
                registered_param_bindings.extend(bindings)


if getattr(shared.opts, "queue_ui_placement", "") != ui_placement_append_to_main:
    script_callbacks.on_ui_tabs(on_ui_tab)

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(on_app_started)
