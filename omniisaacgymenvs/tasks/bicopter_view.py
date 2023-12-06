from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class BicopterView(ArticulationView):
    def __init__(self, prim_paths_expr: str, name: Optional[str] = "BicopterView") -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        self.propeller = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/bicopter/propeller[0-1]", name="propeller_view", reset_xform_properties=False
        )
