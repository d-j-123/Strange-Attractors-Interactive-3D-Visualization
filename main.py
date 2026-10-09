from system import DynamicSystem

system = 'lorentz'

ds = DynamicSystem(system)

values, time = ds.solve_system()

ds.show_system(values, time)

