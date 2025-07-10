import os
import sys
import abc
import atexit
import time
import json
import logging
import platform
import requests
from modules import shared
import traceback
from typing import Callable, List, NoReturn, Any

import gradio as gr
from gradio.blocks import Block, BlockContext

is_windows = platform.system() == "Windows"
is_macos = platform.system() == "Darwin"

if logging.getLogger().hasHandlers():
    log = logging.getLogger("sd")
else:
    import copy
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            "DEBUG": "\033[0;36m",  # CYAN
            "INFO": "\033[0;32m",  # GREEN
            "WARNING": "\033[0;33m",  # YELLOW
            "ERROR": "\033[0;31m",  # RED
            "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
            "RESET": "\033[0m",  # RESET COLOR
        }

        def format(self, record):
            colored_record = copy.copy(record)
            levelname = colored_record.levelname
            seq = self.COLORS.get(levelname, self.COLORS["RESET"])
            colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
            return super().format(colored_record)

    # Create a new logger
    logger = logging.getLogger("AgentScheduler")
    logger.propagate = False

    # Add handler if we don't have one.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter("%(levelname)s - %(message)s"))
        logger.addHandler(handler)

    # Configure logger
    loglevel = logging.INFO
    logger.setLevel(loglevel)

    log = logger

translations = {}
loc_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'localizations', 'zh-CN.json')
if os.path.exists(loc_file):
    with open(loc_file, 'r', encoding='utf-8') as f:
        translations = json.load(f)

def i18n(text):
    lang = getattr(shared.opts, "agent_scheduler_language", "English")
    if lang == "中文" and text in translations:
        return translations[text]
    return text


class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def compare_components_with_ids(components: List[Block], ids: List[int]):
    return len(components) == len(ids) and all(
        component._id == _id for component, _id in zip(components, ids)
    )


def get_component_by_elem_id(root: Block, elem_id: str):
    if root.elem_id == elem_id:
        return root

    elem = None
    if isinstance(root, BlockContext):
        for block in root.children:
            elem = get_component_by_elem_id(block, elem_id)
            if elem is not None:
                break

    return elem


def get_components_by_ids(root: gr.Blocks, ids: List[Any]) -> List[Block]:
    """
    Finds Gradio components in a Blocks root by their ID.
    """
    components = []
    if not isinstance(ids, list):
        return components

    for cid in ids:
        component = None
        if isinstance(cid, int):
            block = root.blocks.get(cid)
            if isinstance(block, Block):
                component = block
        elif isinstance(cid, Block):
            component = cid
        
        if component is not None:
            components.append(component)

    return components


def detect_control_net(root: gr.Blocks, submit: gr.Button):
    UiControlNetUnit = None

    dependencies: List[dict] = [
        x
        for x in root.config['dependencies']
        if x.get("trigger") == "click" and submit._id in x["targets"]
    ]
    for d in dependencies:
        if len(d["outputs"]) == 1:
            outputs = get_components_by_ids(root, d["outputs"])
            output = outputs[0]
            if (
                isinstance(output, gr.State)
                and type(output.value).__name__ == "UiControlNetUnit"
            ):
                UiControlNetUnit = type(output.value)

    return UiControlNetUnit


def get_dict_attribute(dict_inst: dict, name_string: str, default=None):
    nested_keys = name_string.split(".")
    value = dict_inst

    for key in nested_keys:
        value = value.get(key, None)

        if value is None:
            return default

    return value


def set_dict_attribute(dict_inst: dict, name_string: str, value):
    """
    Set an attribute to a dictionary using dot notation.
    If the attribute does not already exist, it will create a nested dictionary.

    Parameters:
        - dict_inst: the dictionary instance to set the attribute
        - name_string: the attribute name in dot notation (ex: 'attribute.name')
        - value: the value to set for the attribute

    Returns:
        None
    """
    # Split the attribute names by dot
    name_list = name_string.split(".")

    # Traverse the dictionary and create a nested dictionary if necessary
    current_dict = dict_inst
    for name in name_list[:-1]:
        if name not in current_dict:
            current_dict[name] = {}
        current_dict = current_dict[name]

    # Set the final attribute to its value
    current_dict[name_list[-1]] = value


def request_with_retry(
    make_request: Callable[[], requests.Response],
    max_try: int = 3,
    retries: int = 0,
):
    try:
        res = make_request()
        if res.status_code > 400:
            raise Exception(res.text)

        return True
    except requests.exceptions.ConnectionError:
        log.error("[ArtVenture] Connection error while uploading result")
        if retries >= max_try - 1:
            return False

        time.sleep(1)
        log.info(f"[ArtVenture] Retrying {retries + 1}...")
        return request_with_retry(
            make_request,
            max_try=max_try,
            retries=retries + 1,
        )
    except Exception as e:
        log.error("[ArtVenture] Error while uploading result")
        log.error(e)
        log.debug(traceback.format_exc())
        return False


def _exit(status: int) -> NoReturn:
    try:
        atexit._run_exitfuncs()
    except:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
