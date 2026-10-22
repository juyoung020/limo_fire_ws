import os

import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from .limo_ros2_diff import LimoROS2Diff

class LimoROS2DifferentialExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "Limo ROS2 Differential"
        self.category = "Wego"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Limo ROS2 Differential",
            "doc_link": "",
            "overview": "You will Learn how to set limo ros2 differential",
            "sample": LimoROS2Diff(),
        }

        ui_handle = BaseSampleUITemplate(**ui_kwargs)

        # register the example with examples browser
        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)

        return
