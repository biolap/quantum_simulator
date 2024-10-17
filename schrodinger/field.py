import ast
import numexpr as ne

class Field:
    """
    Класс для определения потенциала и препятствий в симуляции.
    """
    def __init__(self):
        self.potential_expr = "0"
        self.obstacle_expr = "False"
        self.potential_func = None
        self.obstacle_func = None

    def _compile_expr(self, expr, context):
        """Компилирует выражение в функцию с помощью numexpr."""
        try:
            tree = ast.parse(expr, mode='eval')
            code = compile(tree, filename='<string>', mode='eval')
            return lambda x, y: ne.evaluate(expr, local_dict=context)
        except (SyntaxError, TypeError, NameError) as e:
            print(f"Error compiling expression: {e}")
            return None

    def set_potential(self, expr):
        """Устанавливает выражение для потенциала."""
        self.potential_expr = expr
        self.potential_func = self._compile_expr(expr, {'x': 0, 'y': 0})

    def set_obstacle(self, expr):
        """Устанавливает выражение для препятствий."""
        self.obstacle_expr = expr
        self.obstacle_func = self._compile_expr(expr, {'x': 0, 'y': 0})

    def is_obstacle(self, x, y):
        """Проверяет, является ли точка препятствием."""
        if self.obstacle_func:
            return bool(self.obstacle_func(x, y))
        return False

    def get_potential(self, x, y):
        """Возвращает значение потенциала в точке (x, y)."""
        if self.potential_func:
            return complex(self.potential_func(x, y))
        return 0j