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

    def _compile_expr(self, expr):
        """Компилирует выражение в функцию с помощью numexpr."""
        try:
            ast.parse(expr, mode='eval')
            return expr
        except (SyntaxError, TypeError, NameError) as e:
            print(f"Error compiling expression: {e}")
            return None

    def set_potential(self, expr):
        """Устанавливает выражение для потенциала."""
        self.potential_expr = expr
        self.potential_func = self._compile_expr(expr)

    def set_obstacle(self, expr):
        """Устанавливает выражение для препятствий."""
        self.obstacle_expr = expr
        self.obstacle_func = self._compile_expr(expr)

    def is_obstacle(self, x, y):
        """Проверяет, является ли точка препятствием."""
        if self.obstacle_func:
            return bool(ne.evaluate(self.obstacle_func, local_dict={'x': x, 'y': y}))
        return False

    def get_potential(self, x, y, t):  #  <--- Добавлен параметр t
        if self.potential_func:
            return complex(ne.evaluate(self.potential_func, local_dict={'x': x, 'y': y, 't': t})) # <--- Используем t
        return 0j