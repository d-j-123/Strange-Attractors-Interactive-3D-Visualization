import numpy as np
from solver import RK4
import pyvista as pv

class DynamicSystem:
    def __init__(self, system):
        self.system = system
        self.parameters = self.get_parameters()
        self.equations = self.get_equations()
        self.initial_conditions = self.get_initial_conditions()

    def get_parameters(self):
        if self.system == 'lorentz':
            #Initial values of parameters
            r = 28
            sigma = 10
            b = 8 / 3
            
            parameters = {"r": {"value": r, "range": (0.0, 50.0)}, "sigma": {"value": sigma, "range": (0.0, 20.0)}, "b": {"value": b, "range": (0.0, 5.0)}}

        elif self.system == 'rossler':
            #Initial values of parameters
            a = 0.2
            b = 0.2
            c = 14

            parameters = {"a": {"value": a, "range": (0.0, 0.5)}, "b": {"value": b, "range": (0.0, 0.5)}, "c": {"value": c, "range": (0.0, 25.0)}}

        elif self.system == 'thomas':
            #Initial value of parameter
            b = 0.3

            parameters = {"b": {"value": b, "range": (0.0, 1.0)}}

        elif self.system == 'aizawa':
            #Initial values of parameters
            a = 0.95
            b = 0.7
            c = 0.6
            d = 3.5
            e = 0.25
            g = 0.1
        
            parameters = {"a": {"value": a, "range": (0.0, 1.5)}, "b": {"value": b, "range": (0.0, 1.5)}, "c": {"value": c, "range": (0.0, 1.5)}, "d": {"value": d, "range": (0.0, 6.0)}, "e": {"value": e, "range": (0.0, 0.5)}, "g": {"value": g, "range": (-0.2, 0.2)}}

        elif self.system == 'halvorsen':
            #Initial value of parameter
            a = 1.4
        
            parameters = {"a": {"value": a, "range": (0.0, 3.0)}}

        elif self.system == 'chen':
            #Initial values of parameters
            a = 40
            b = 3
            c = 28
        
            parameters = {"a": {"value": a, "range": (0.0, 50.0)}, "b": {"value": b, "range": (0.0, 10.0)}, "c": {"value": c, "range": (0.0, 40.0)}}

        return parameters

    def get_initial_conditions(self):
        x0, y0, z0 = 1, 1, 1

        initial_conditions = {"x0": {"value": x0, "range": (-10.0, 10.0)}, "y0": {"value": y0, "range": (-10.0, 10.0)}, "z0": {"value": z0, "range": (-10.0, 10.0)}}

        return initial_conditions

    def get_equations(self):
        if self.system == 'lorentz':
            self.name = "Lorentz attractor"

            r = self.parameters["r"]["value"]
            sigma = self.parameters["sigma"]["value"]
            b = self.parameters["b"]["value"]

            equations = lambda t, f: np.array([sigma * (f[1] - f[0]), f[0] * (r - f[2]) - f[1], f[0] * f[1] - b * f[2]])

        elif self.system == 'rossler':
            self.name = "Rossler attractor"

            a = self.parameters["a"]["value"]
            b = self.parameters["b"]["value"]
            c = self.parameters["c"]["value"]

            equations = lambda t, f: np.array([-f[1] - f[2], f[0] + a * f[1], b + f[2] * (f[0] - c)])

        elif self.system == 'thomas':
            self.name = 'Thomas attractor'

            b = self.parameters["b"]["value"]

            equations = lambda t, f: np.array([np.sin(f[1]) - b * f[0], np.sin(f[2]) - b * f[1], np.sin(f[0]) - b * f[2]])

        elif self.system == 'aizawa':
            self.name = 'Aizawa attractor'

            a = self.parameters["a"]["value"]
            b = self.parameters["b"]["value"]
            c = self.parameters["c"]["value"]
            d = self.parameters["d"]["value"]
            e = self.parameters["e"]["value"]
            g = self.parameters["g"]["value"]

            equations = lambda t, f: np.array([(f[2] - b) * f[0] - d * f[1], d * f[0] + (f[2] - b) * f[1], c + a * f[2] - f[2] ** 3 / 3 - (f[0] ** 2 + f[1] ** 2) * (1 + e * f[2]) + g * f[2] * f[0] ** 3])

        elif self.system == 'halvorsen':
            self.name = 'Halvorsen attractor'

            a = self.parameters["a"]["value"]

            equations = lambda t, f: np.array([- a * f[0] - 4 * f[1] - f[2] ** 2, - a * f[1] - 4 * f[2] - f[0] ** 2, - a * f[2] - 4 * f[0] - f[1] ** 2])

        elif self.system == 'chen':
            self.name = 'Chen attractor'

            a = self.parameters["a"]["value"]
            b = self.parameters["b"]["value"]
            c = self.parameters["c"]["value"]

            equations = lambda t, f: np.array([a * (f[1] - f[0]), (c - a) * f[0] - f[0] * f[2] + c * f[1], f[0] * f[1] - b * f[2]])

        return equations

    def solve_system(self):
        initial_conditions = np.array([param["value"] for param in self.initial_conditions.values()])
        values, time = RK4(f=self.equations, f_init=initial_conditions, t0=0, t_end=50, h=0.001)

        return values, time

    def show_system(self, values, time):
        plotter = pv.Plotter()

        mesh = pv.lines_from_points(values)
        mesh['time'] = np.linspace(0, 1, len(values))

        engine = MyCustomRoutine(self, mesh, plotter)
        #plotter.add_mesh(mesh, color='yellow_green')
        plotter.add_mesh(mesh, scalars='time', cmap='cool', line_width=3, render_lines_as_tubes=True, show_scalar_bar=False)
        plotter.background_color = 'black'

        plotter.add_light(pv.Light(position=(10, 10, 20), color='cyan', intensity=0.8))
        plotter.add_light(pv.Light(position=(-10, -5, 10), color='magenta', intensity=0.5))

        n = len(self.parameters)
        cols = min(n, 3)

        #param sliders
        for i, (key, param) in enumerate(self.parameters.items()):
            col = i % cols
            row = i // cols

            x_start = 0.05 + col * 0.32
            x_end = x_start + 0.27
            y = 0.1 + row * 0.125

            plotter.add_slider_widget(
                callback=lambda value, key=key: engine(key, value), rng=param["range"], value=param["value"], title=key, pointa=(x_start, y), pointb=(x_end, y), style='modern', color="white"
            )

        m = len(self.initial_conditions)
        cols = min(m, 3)

        #init conditions sliders
        for j, (key, param) in enumerate(self.initial_conditions.items()):
            col = j % cols
            row = j // cols
            
            x_start = 0.05 + col * 0.32
            x_end = x_start + 0.27
            y = 0.9 + row * 0.125

            plotter.add_slider_widget(
                callback=lambda value, key=key: engine(key, value, is_initial=True), rng=param["range"], value=param["value"], title=key, pointa=(x_start, y), pointb=(x_end, y), style='modern', color="white"
            )

        plotter.show(title=self.name)


class MyCustomRoutine:
    """Stateful callback for updating a mesh from slider parameters."""

    def __init__(self, system, mesh, plotter):
        self.mesh = mesh  # Expected PyVista mesh type
        self.plotter = plotter
        self.system = system

    def __call__(self, param, value, is_initial=False):
        if is_initial:
            self.system.initial_conditions[param]["value"] = value
        else:  
            self.system.parameters[param]["value"] = value
            self.system.equations = self.system.get_equations()

        values, time = self.system.solve_system()
        new_mesh = pv.lines_from_points(values)
        new_mesh["time"] = np.linspace(0, 1, len(values))

        self.mesh.copy_from(new_mesh)
        self.plotter.render()
