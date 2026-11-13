import unittest
from eyecomm.server.store import DEFAULT_TILES, validate_tiles

class BoardValidationTests(unittest.TestCase):
    def test_default_board_is_valid(self):
        self.assertTrue(validate_tiles(DEFAULT_TILES))

    def test_missing_tile_is_invalid(self):
        self.assertFalse(validate_tiles(DEFAULT_TILES[:-1]))

    def test_action_cannot_be_changed(self):
        tiles = [dict(tile) for tile in DEFAULT_TILES]
        tiles[3]["action"] = "WORD"
        self.assertFalse(validate_tiles(tiles))

if __name__ == "__main__":
    unittest.main()
