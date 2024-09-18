import numpy as np
import pybullet as p
from pybullet_envs.minitaur.envs import env_randomizer_base

class HeightfieldRandomizer(env_randomizer_base.EnvRandomizerBase):
    """Generates an uneven terrain in the gym env with a solid color."""

    def __init__(self, 
                 max_height_perturbation=1, 
                 min_height_perturbation=-0.0, 
                 solid_color=(0.5, 0.5, 0.5, 1),  # Set a default solid gray color
                 n_rows=100, 
                 n_cols=100,
                 roughness=0.01,
                 cell_size=1):
        """Initializes the randomizer.

        Args:
          max_height_perturbation: Max height of bumps in meters.
          min_height_perturbation: Min height of bumps in meters.
          solid_color: A tuple representing the RGBA color of the ground.
          n_rows: Number of rows in the heightfield grid.
          n_cols: Number of columns in the heightfield grid.
          roughness: Roughness of the terrain, affecting the bumpiness.
          cell_size: Size of each grid cell in meters; controls bump size.
        """
        self._max_height_perturbation = max_height_perturbation
        self._min_height_perturbation = min_height_perturbation
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._roughness = roughness
        self._cell_size = cell_size
        self._heightfield_data = [0] * self._n_rows * self._n_cols
        self._terrain_shape = -1
        self._initial = True
        self.terrain = None
        self.solid_color = solid_color

    def randomize_env(self, env):
        # Create a heightfield with larger grid to simulate roughness
        larger_n_rows = int(self._n_rows * 2)
        larger_n_cols = int(self._n_cols * 2)
        larger_heightfield_data = [0] * (larger_n_rows * larger_n_cols)

        for j in range(larger_n_rows):
            for i in range(larger_n_cols):
                height = np.random.uniform(self._min_height_perturbation, self._max_height_perturbation)
                if np.random.rand() < self._roughness:
                    height += np.random.uniform(0, self._max_height_perturbation * 0.5)
                
                larger_heightfield_data[i + j * larger_n_cols] = height

        # Sample down to the required grid size
        for j in range(self._n_rows):
            for i in range(self._n_cols):
                x = int(i * (larger_n_cols / self._n_cols))
                y = int(j * (larger_n_rows / self._n_rows))
                self._heightfield_data[i + j * self._n_cols] = larger_heightfield_data[x + y * larger_n_cols]

        # Disable rendering temporarily to speed up terrain update
        if env.rendering_enabled:
            env.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # Create or update the heightfield collision shape
        self._terrain_shape = env.pybullet_client.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            flags=p.GEOM_CONCAVE_INTERNAL_EDGE,
            meshScale=[self._cell_size, self._cell_size, 1.0],  # Scale in X and Y for bump size
            heightfieldData=self._heightfield_data,
            numHeightfieldRows=self._n_rows,
            numHeightfieldColumns=self._n_cols,
            replaceHeightfieldIndex=self._terrain_shape)

        # If this is the first time setting up the terrain
        if self._initial:
            env.pybullet_client.removeBody(env.get_ground())
            self.terrain = env.pybullet_client.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=self._terrain_shape,
                basePosition=[0, 0, 0])  # Ensure terrain is placed at the origin
            env.set_ground(self.terrain)
            self._initial = False

            # Apply solid color to the terrain visual shape
            env.pybullet_client.changeVisualShape(
                self.terrain, -1, rgbaColor=self.solid_color)

        # Debugging: Print terrain info
        terrain_pos, terrain_orn = env.pybullet_client.getBasePositionAndOrientation(self.terrain)
        print(f"Terrain Position: {terrain_pos}")
        print(f"Terrain Orientation: {terrain_orn}")

        # Center terrain under the robot if needed
        x, y, _ = env.robot.GetBasePosition()
        env.pybullet_client.resetBasePositionAndOrientation(self.terrain, [x, y, 0],
                                                            [0, 0, 0, 1])

        # Re-enable rendering
        if env.rendering_enabled:
            env.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Additional debug visuals
        env.pybullet_client.addUserDebugLine(
            lineFromXYZ=[-self._n_cols * self._cell_size / 2, -self._n_rows * self._cell_size / 2, 0],
            lineToXYZ=[self._n_cols * self._cell_size / 2, self._n_rows * self._cell_size / 2, 0],
            lineColorRGB=[1, 0, 0],
            lineWidth=1.0)

