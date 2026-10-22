import os

import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from .limo_ros2_diff_obs import LimoROS2DiffObs

class LimoROS2DifferentialExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "Limo ROS2 Differential with Obstacle"
        self.category = "Wego"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Limo ROS2 Differential with Obstacle",
            "doc_link": "",
            "overview": "You can test Limo differential mode with obstacles",
            "sample": LimoROS2DiffObs(),
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
