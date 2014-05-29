from functools import partial

chance_of_a_four = .1

__version__ = '1.3.0'

from random import choice, random, randrange, uniform
from copy import copy

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, OptionProperty, ObjectProperty
from kivy.graphics import Color, BorderImage
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.metrics import dp
from kivy.animation import Animation
from kivy.utils import get_color_from_hex
from kivy.core.window import Window
from kivy.utils import platform
from kivy.factory import Factory

platform = platform()
app = None

if platform == 'android':
    # Support for Google Play
    import gs_android

    leaderboard_highscore = 'CgkI0InGg4IYEAIQBg'
    achievement_block_32 = 'CgkI0InGg4IYEAIQCg'
    achievement_block_64 = 'CgkI0InGg4IYEAIQCQ'
    achievement_block_128 = 'CgkI0InGg4IYEAIQAQ'
    achievement_block_256 = 'CgkI0InGg4IYEAIQAg'
    achievement_block_512 = 'CgkI0InGg4IYEAIQAw'
    achievement_block_1024 = 'CgkI0InGg4IYEAIQBA'
    achievement_block_2048 = 'CgkI0InGg4IYEAIQBQ'
    achievement_block_4096 = 'CgkI0InGg4IYEAIQEg'
    achievement_100x_block_512 = 'CgkI0InGg4IYEAIQDA'
    achievement_1000x_block_512 = 'CgkI0InGg4IYEAIQDQ'
    achievement_100x_block_1024 = 'CgkI0InGg4IYEAIQDg'
    achievement_1000x_block_1024 = 'CgkI0InGg4IYEAIQDw'
    achievement_10x_block_2048 = 'CgkI0InGg4IYEAIQEA'
    achievements = {
        32: achievement_block_32,
        64: achievement_block_64,
        128: achievement_block_128,
        256: achievement_block_256,
        512: achievement_block_512,
        1024: achievement_block_1024,
        2048: achievement_block_2048,
        4096: achievement_block_4096}

    from kivy.uix.popup import Popup

    class GooglePlayPopup(Popup):
        pass

else:
    achievements = {}

from kivy.uix.popup import Popup


class AITempoPopup(Popup):
    pass


tempo = .2


class ButtonBehavior(object):
    # XXX this is a port of the Kivy 1.8.0 version, the current android version
    # still use 1.7.2. This is going to be removed soon.
    state = OptionProperty('normal', options=('normal', 'down'))
    last_touch = ObjectProperty(None)

    def __init__(self, **kwargs):
        self.register_event_type('on_press')
        self.register_event_type('on_release')
        super(ButtonBehavior, self).__init__(**kwargs)

    def _do_press(self):
        self.state = 'down'

    def _do_release(self):
        self.state = 'normal'

    def on_touch_down(self, touch):
        if super(ButtonBehavior, self).on_touch_down(touch):
            return True
        if touch.is_mouse_scrolling:
            return False
        if not self.collide_point(touch.x, touch.y):
            return False
        if self in touch.ud:
            return False
        touch.grab(self)
        touch.ud[self] = True
        self.last_touch = touch
        self._do_press()
        self.dispatch('on_press')
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            return True
        if super(ButtonBehavior, self).on_touch_move(touch):
            return True
        return self in touch.ud

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return super(ButtonBehavior, self).on_touch_up(touch)
        assert (self in touch.ud)
        touch.ungrab(self)
        self.last_touch = touch
        self._do_release()
        self.dispatch('on_release')
        return True

    def on_press(self):
        pass

    def on_release(self):
        pass


class Number(Widget):
    number = NumericProperty(2)
    scale = NumericProperty(.1)
    colors = {
        2: get_color_from_hex('#eee4da'),
        4: get_color_from_hex('#ede0c8'),
        8: get_color_from_hex('#f2b179'),
        16: get_color_from_hex('#f59563'),
        32: get_color_from_hex('#f67c5f'),
        64: get_color_from_hex('#f65e3b'),
        128: get_color_from_hex('#edcf72'),
        256: get_color_from_hex('#edcc61'),
        512: get_color_from_hex('#edc850'),
        1024: get_color_from_hex('#edc53f'),
        2048: get_color_from_hex('#edc22e'),
        4096: get_color_from_hex('#ed702e'),
        8192: get_color_from_hex('#ed4c2e')}

    def __init__(self, **kwargs):
        super(Number, self).__init__(**kwargs)
        anim = Animation(scale=1., d=.15, t='out_quad')
        anim.bind(on_complete=self.clean_canvas)
        anim.start(self)

    def clean_canvas(self, *args):
        self.canvas.before.clear()
        self.canvas.after.clear()

    def move_to_and_destroy(self, pos):
        self.destroy()
        #anim = Animation(opacity=0., d=.25, t='out_quad')
        #anim.bind(on_complete=self.destroy)
        #anim.start(self)

    def destroy(self, *args):
        self.parent.remove_widget(self)

    def move_to(self, pos):
        if self.pos == pos:
            return
        Animation(pos=pos, d=.1, t='out_quad').start(self)

    def on_number(self, instance, value):
        if platform == 'android':
            if value in achievements:
                app.gs_unlock(achievements[value])
            if value == 512:
                app.gs_increment(achievement_100x_block_512)
                app.gs_increment(achievement_1000x_block_512)
            elif value == 1024:
                app.gs_increment(achievement_100x_block_1024)
                app.gs_increment(achievement_1000x_block_1024)
            elif value == 2048:
                app.gs_increment(achievement_10x_block_2048)


class Game2048(Widget):
    cube_size = NumericProperty(10)
    cube_padding = NumericProperty(10)
    score = NumericProperty(0)

    def __init__(self, **kwargs):
        super(Game2048, self).__init__()
        self.grid = [
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None]]

        # bind keyboard
        Window.bind(on_key_down=self.on_key_down)
        Window.on_keyboard = lambda *x: None

        self.restart()

    def on_key_down(self, window, key, *args):
        if key == 273:
            self.move_topdown(True)
        elif key == 274:
            self.move_topdown(False)
        elif key == 276:
            self.move_leftright(False)
        elif key == 275:
            self.move_leftright(True)
        elif key == 27 and platform == 'android':
            from jnius import autoclass

            PythonActivity = autoclass('org.renpy.android.PythonActivity')
            PythonActivity.mActivity.moveTaskToBack(True)
            return True


    def rebuild_background(self):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0xbb / 255., 0xad / 255., 0xa0 / 255.)
            BorderImage(pos=self.pos, size=self.size, source='data/round.png')
            Color(0xcc / 255., 0xc0 / 255., 0xb3 / 255.)
            csize = self.cube_size, self.cube_size
            for ix, iy in self.iterate_pos():
                BorderImage(pos=self.index_to_pos(ix, iy), size=csize,
                            source='data/round.png')


    def reposition(self, *args):
        self.rebuild_background()
        # calculate the size of a number
        l = min(self.width, self.height)
        padding = (l / 4.) / 8.
        cube_size = (l - (padding * 5)) / 4.
        self.cube_size = cube_size
        self.cube_padding = padding

        for ix, iy, number in self.iterate():
            number.size = cube_size, cube_size
            number.pos = self.index_to_pos(ix, iy)

    def iterate(self):
        for ix, iy in self.iterate_pos():
            child = self.grid[ix][iy]
            if child:
                yield ix, iy, child

    def iterate_empty(self):
        for ix, iy in self.iterate_pos():
            child = self.grid[ix][iy]
            if not child:
                yield ix, iy

    def iterate_pos(self):
        for ix in range(4):
            for iy in range(4):
                yield ix, iy

    def index_to_pos(self, ix, iy):
        padding = self.cube_padding
        cube_size = self.cube_size
        return [
            (self.x + padding) + ix * (cube_size + padding),
            (self.y + padding) + iy * (cube_size + padding)]

    def spawn_number(self, *args):
        empty = list(self.iterate_empty())
        if not empty:
            return
        value = 4 if random() < chance_of_a_four else 2
        ix, iy = choice(empty)
        self.spawn_number_at(ix, iy, value)

    def spawn_number_at(self, ix, iy, value):
        number = Number(
            size=(self.cube_size, self.cube_size),
            pos=self.index_to_pos(ix, iy),
            number=value)
        self.grid[ix][iy] = number
        self.add_widget(number)

    def on_touch_up(self, touch):
        v = Vector(touch.pos) - Vector(touch.opos)
        if v.length() < dp(20):
            return

        # detect direction
        dx, dy = v
        if abs(dx) > abs(dy):
            self.move_leftright(dx > 0)
        else:
            self.move_topdown(dy > 0)
        return True

    def move_leftright(self, right):
        r = range(3, -1, -1) if right else range(4)
        grid = self.grid
        moved = False

        for iy in range(4):
            # get all the cube for the current line
            cubes = []
            for ix in r:
                cube = grid[ix][iy]
                if cube:
                    cubes.append(cube)

            # combine them
            self.combine(cubes)

            # update the grid
            for ix in r:
                cube = cubes.pop(0) if cubes else None
                if grid[ix][iy] != cube:
                    moved = True
                grid[ix][iy] = cube
                if not cube:
                    continue
                pos = self.index_to_pos(ix, iy)
                if cube.pos != pos:
                    cube.move_to(pos)

        if not self.check_end() and moved:
            Clock.schedule_once(self.spawn_number, tempo)

    def move_topdown(self, top):
        r = range(3, -1, -1) if top else range(4)
        grid = self.grid
        moved = False

        for ix in range(4):
            # get all the cube for the current line
            cubes = []
            for iy in r:
                cube = grid[ix][iy]
                if cube:
                    cubes.append(cube)

            # combine them
            self.combine(cubes)

            # update the grid
            for iy in r:
                cube = cubes.pop(0) if cubes else None
                if grid[ix][iy] != cube:
                    moved = True
                grid[ix][iy] = cube
                if not cube:
                    continue
                pos = self.index_to_pos(ix, iy)
                if cube.pos != pos:
                    cube.move_to(pos)

        if not self.check_end() and moved:
            Clock.schedule_once(self.spawn_number, tempo)

    def combine(self, cubes):
        if len(cubes) <= 1:
            return cubes
        index = 0
        while index < len(cubes) - 1:
            cube1 = cubes[index]
            cube2 = cubes[index + 1]
            if cube1.number == cube2.number:
                cube1.number *= 2
                self.score += cube1.number
                cube2.move_to_and_destroy(cube1.pos)
                del cubes[index + 1]

            index += 1

    def check_end(self):
        # we still have empty space
        if any(self.iterate_empty()):
            return False

        # check if 2 numbers of the same type are near each others
        if self.have_available_moves():
            return False

        self.end()
        return True

    def have_available_moves(self):
        grid = self.grid
        for iy in range(4):
            for ix in range(3):
                cube1 = grid[ix][iy]
                cube2 = grid[ix + 1][iy]
                if cube1.number == cube2.number:
                    return True

        for ix in range(4):
            for iy in range(3):
                cube1 = grid[ix][iy]
                cube2 = grid[ix][iy + 1]
                if cube1.number == cube2.number:
                    return True

    def end(self):
        end = self.ids.end.__self__
        self.remove_widget(end)
        self.add_widget(end)
        text = 'Game\nover!'
        for ix, iy, cube in self.iterate():
            if cube.number == 2048:
                text = 'WIN !'
        self.ids.end_label.text = text
        Animation(opacity=1., d=.5).start(end)
        app.gs_score(self.score)

    def restart(self):
        self.score = 0
        for ix, iy, child in self.iterate():
            child.destroy()
        self.grid = [
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None]]
        self.reposition()
        Clock.schedule_once(self.spawn_number, .1)
        Clock.schedule_once(self.spawn_number, .1)
        self.ids.end.opacity = 0


class Game2048App(App):
    use_kivy_settings = False
    ai_api = None

    def build_config(self, config):
        if platform == 'android':
            config.setdefaults('play', {'use_google_play': '0'})

    def build(self):
        global app
        app = self

        if platform == 'android':
            self.use_google_play = self.config.getint('play', 'use_google_play')
            if self.use_google_play:
                gs_android.setup(self)
            else:
                Clock.schedule_once(self.ask_google_play, .5)
        else:
            # remove all the leaderboard and achievement buttons
            scoring = self.root.ids.scoring
            scoring.parent.remove_widget(scoring)

        self.ai_api = Game2048AI(self.root.ids.game)
        self.root.ids.tempo_controls.opacity = 0

    def gs_increment(self, uid):
        if platform == 'android' and self.use_google_play:
            gs_android.increment(uid, 1)

    def gs_unlock(self, uid):
        if platform == 'android' and self.use_google_play:
            gs_android.unlock(uid)

    def gs_score(self, score):
        if platform == 'android' and self.use_google_play:
            gs_android.leaderboard(leaderboard_highscore, score)

    def gs_show_achievements(self):
        if platform == 'android':
            if self.use_google_play:
                gs_android.show_achievements()
            else:
                self.ask_google_play()

    def gs_show_leaderboard(self):
        if platform == 'android':
            if self.use_google_play:
                gs_android.show_leaderboard(leaderboard_highscore)
            else:
                self.ask_google_play()

    def ask_google_play(self, *args):
        popup = GooglePlayPopup()
        popup.open()

    def activate_google_play(self):
        self.config.set('play', 'use_google_play', '1')
        self.config.write()
        self.use_google_play = 1
        gs_android.setup(self)

    def on_pause(self):
        if platform == 'android':
            gs_android.on_stop()
        return True

    def on_resume(self):
        if platform == 'android':
            gs_android.on_start()

    def _on_keyboard_settings(self, *args):
        return

    #**************************************************************************
    # Executa uma jogada
    def ai_move(self):
        return self.ai_api.make_move()

    # Comeca a jogar continuamente ou, se ja estiver jogando, para de jogar
    def ai_play(self, button):
        if button.state == 'down':
            button.text = 'Parar'
            self.ai_play_button = button
            Animation(opacity=1., d=.5).start(self.root.ids.tempo_controls)
            Clock.schedule_interval(self.ai_keep_playing, tempo)
        else:
            self.ai_stop_playing()

    def ai_keep_playing(self, dt):
        if not self.ai_move():
            self.ai_stop_playing()
            return False  # automatically unschedule callback

    def ai_stop_playing(self):
        self.ai_play_button.state = 'normal'
        self.ai_play_button.text = 'Jogar'
        self.root.ids.tempo_controls.opacity = 0
        Clock.unschedule(self.ai_keep_playing)

    def ai_decrease_tempo(self):
        global tempo
        tempo -= .03
        Clock.unschedule(self.ai_keep_playing)
        Clock.schedule_interval(self.ai_keep_playing, tempo)

    def ai_increase_tempo(self):
        global tempo
        tempo += .03
        Clock.unschedule(self.ai_keep_playing)
        Clock.schedule_interval(self.ai_keep_playing, tempo)

        #**************************************************************************


#**************************************************************************
class Actions:
    # up, down, left, right = range(4)
    up, down, left, right = ("UP", "DOWN", "LEFT", "RIGHT")


#----------------
class State(object):
    board = None

    def __init__(self, grid):
        self.board = []
        for iy in range(4):
            row = []
            for ix in range(4):
                number = 0
                cube = grid[ix][iy]
                if cube:
                    number = cube.number
                row.append(number)
            self.board.append(row)

    def __copy__(self):
        # newone = type(self)()
        # newone.board = [row[:] for row in self.board]
        # return newone
        cls = self.__class__
        result = cls.__new__(cls)
        result.board = [row[:] for row in self.board]
        # result.__dict__.update(self.__dict__)
        return result

    # def __iter__(self):
    #     for x in list.__iter__(self):
    #         yield self.do_something(x)

    def __iter__(self):
        for iy, row in enumerate(self.board):
            for ix, value in enumerate(row):
                yield ix, iy, value

    def __str__(self):
        rows_str = [' '.join(['{:^4}'.format(number) for number in row]) for row in self.board]
        string = '\n'.join(reversed(rows_str))
        return string

    def value_at(self, x, y):
        return self.board[y][x]  # y is the row and x, the column

    def set_value(self, x, y, value):
        self.board[y][x] = value  # y is the row and x, the column

    @staticmethod
    def iterate_value(state, value):
        for iy, row in enumerate(state):
            for ix, val in enumerate(row):
                if val == value:
                    yield ix, iy, val

    def next_state(self, action):
        state = copy(self)  # make a copy
        if action is Actions.up:
            state.combine_up()
        elif action is Actions.down:
            state.combine_down()
        elif action is Actions.right:
            state.combine_right()
        elif action is Actions.left:
            state.combine_left()
        return state

    def generate_chance_states(self, state):
        """Gera os possiveis estados onde um 2 ou um 4 pode aparecer e retorna uma lista de pares (chance, estado).
        @param state:
        @type state:
        @return: A list of tuples (chance, state).
        @rtype:
        """
        # empty = list(self.iterate_value(state, 0))
        empty = [(x, y) for x, y, value in self if value == 0]  # get empty blocks
        if not empty:
            return [(1.0, copy(state))]
        chance_states = []
        total_states = float(len(empty)) * 2.0  # Two possibilities for each empty position: a 2 or a 4
        for x, y in empty:
            state2 = copy(state)
            state2.set_value(x, y, 2)
            chance2 = (1.0 - chance_of_a_four) * 1.0 / total_states
            chance_states.append((chance2, state2))
            state4 = copy(state)
            state4.set_value(x, y, 4)
            chance4 = (chance_of_a_four) * 1.0 / total_states
            chance_states.append((chance4, state4))
        return chance_states

    def combine_down(self):
        for j in range(4):
            blocks = []
            for i in range(4):
                block = self.board[i][j]
                if block != 0:
                    blocks.append(block)

            # combine them
            self.combine(blocks)

            # update the grid
            for i in range(4):
                block = blocks.pop(0) if blocks else 0
                self.board[i][j] = block

    def combine_up(self):
        for j in range(4):
            blocks = []
            for i in range(3, -1, -1):
                block = self.board[i][j]
                if block != 0:
                    blocks.append(block)

            # combine them
            self.combine(blocks)

            # update the grid
            for i in range(3, -1, -1):
                block = blocks.pop(0) if blocks else 0
                self.board[i][j] = block

    def combine_left(self):
        for i in range(4):
            blocks = []
            for j in range(4):
                block = self.board[i][j]
                if block != 0:
                    blocks.append(block)

            # combine them
            self.combine(blocks)

            # update the grid
            for j in range(4):
                block = blocks.pop(0) if blocks else 0
                self.board[i][j] = block

    def combine_right(self):
        for i in range(4):
            blocks = []
            for j in range(3, -1, -1):
                block = self.board[i][j]
                if block != 0:
                    blocks.append(block)

            # combine them
            self.combine(blocks)

            # update the grid
            for j in range(3, -1, -1):
                block = blocks.pop(0) if blocks else 0
                self.board[i][j] = block

    @staticmethod
    def combine(blocks):
        if len(blocks) <= 1:
            return blocks
        i = 0
        while i < len(blocks) - 1:
            block1 = blocks[i]
            block2 = blocks[i + 1]
            if block1 == block2:
                blocks[i] *= 2
                #self.score += block1
                del blocks[i + 1]
            i += 1

    def get_actions(self):
        available = set()
        for ix, iy, value in self:
            if len(available) == 4:  # all actions already available
                break
            if value == 0:  # empty position
                try:
                    if self.value_at(ix + 1, iy) != 0:  # look at right block
                        available.add(Actions.left)
                except IndexError:
                    pass
                try:
                    if self.value_at(ix, iy + 1) != 0:  # look at block above
                        available.add(Actions.down)
                except IndexError:
                    pass
            else:  # position with a number
                try:
                    right_value = self.value_at(ix + 1, iy)  # look at right block
                    if right_value == 0:
                        available.add(Actions.right)
                    if value == right_value:
                        available.add(Actions.right)
                        available.add(Actions.left)
                except IndexError:
                    pass
                try:
                    above_value = self.value_at(ix, iy + 1)  # look at block above
                    if above_value == 0:
                        available.add(Actions.up)
                    if value == above_value:
                        available.add(Actions.up)
                        available.add(Actions.down)
                except IndexError:
                    pass
        return list(available)

    def choose_action(self, eval_func):
        max_score = 0
        max_action = None
        for action in self.get_actions():
            # print("Current state:")
            # print(state)
            # print("\n----------")
            next_state = self.next_state(action)
            score = eval_func(next_state)
            if score > max_score:
                max_score = score
                max_action = action
        #     print("Action: " + action)
        #     print("Next state:")
        #     print(state.next_state(action))
        #     print("Score: " + str(score))
        #     print("\n----------")
        # print("MAX Action: " + max_action)
        # print("Next state:")
        # print(state.next_state(max_action))
        return max_action

    def get_scores_and_actions(self, eval_func, levels_to_go=0):
        """
        @param eval_func:
        @type eval_func:
        @return: A list of tuples (score, action) for all possible actions.
        @rtype:
        """
        scores_actions = [(0.0, None)]
        for action in self.get_actions():
            next_state = self.next_state(action)
            chance_states = self.generate_chance_states(next_state)
            total_score = 0.0
            for chance, state in chance_states:
                #-- Expectimax! --#
                score = 0
                if levels_to_go == 0:
                    score = eval_func(state)
                else:  # recurse!
                    sub_scores = state.get_scores_and_actions(eval_func, levels_to_go-1)
                    score, tmp = max(sub_scores)
                #-----------------#
                chance_score = score * chance
                total_score += chance_score
            scores_actions.append((total_score, action))
        return scores_actions

    def choose_action42(self, eval_func):
        max_score, max_action = max(self.get_scores_and_actions(eval_func))
        return max_action

    def deep_choose_action42(self, eval_func, levels_to_go):
        max_score, max_action = max(self.get_scores_and_actions(eval_func, levels_to_go))
        return max_action
        # if levels_to_go == 0:  # so current state (self) is a leaf; evaluate it
        #     return eval_func(self)


#----------------
class Game2048AI:
    game_app = None

    def __init__(self, game_app):
        self.game_app = game_app

    def current_state(self):
        return State(self.game_app.grid)

    def execute(self, action):
        if action == Actions.up:
            self.game_app.move_topdown(True)
        elif action == Actions.down:
            self.game_app.move_topdown(False)
        elif action == Actions.right:
            self.game_app.move_leftright(True)
        elif action == Actions.left:
            self.game_app.move_leftright(False)

    def make_move(self):
        state = self.current_state()
        action = state.deep_choose_action42(eval_highest_block, 2)
        if not action:
            self.execute(Actions.up)  # just to make the game end
            return False
        self.execute(action)
        return True


#**************************************************************************

def eval_highest_block(state):
    return max([val for x, y, val in state])


def eval_sum_blocks(state):
    return sum([val for x, y, val in state])

#**************************************************************************
if __name__ == '__main__':
    Factory.register('ButtonBehavior', cls=ButtonBehavior)
    Game2048App().run()
    

