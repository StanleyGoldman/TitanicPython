import main
from unittest import TestCase


class Test_extract_cabin_floor(TestCase):
    def test_none(self):
        self.assertEqual(main.extract_cabin_floor(None), None)

    def test_empty(self):
        self.assertEqual(main.extract_cabin_floor(""), None)

    def test_single_floor(self):
        self.assertEqual(main.extract_cabin_floor("C"), "C")

    def test_single_floor_duplicates(self):
        self.assertEqual(main.extract_cabin_floor("C C"), "C")
        self.assertEqual(main.extract_cabin_floor("C C C"), "C")

    def test_multiple_floors(self):
        self.assertEqual(main.extract_cabin_floor("C D"), "CD")

    def test_multiple_floors_duplicated(self):
        self.assertEqual(main.extract_cabin_floor("C C D"), "CD")

    def test_single_room(self):
        self.assertEqual(main.extract_cabin_floor("C45"), "C")

    def test_multiple_room_same_floor(self):
        self.assertEqual(main.extract_cabin_floor("C45 C46"), "C")

    def test_multiple_room_different_floors(self):
        self.assertEqual(main.extract_cabin_floor("C45 D46"), "CD")

    def test_floor_and_room_same_floor(self):
        self.assertEqual(main.extract_cabin_floor("C C45"), "C")
        self.assertEqual(main.extract_cabin_floor("C45 C"), "C")

    def test_floor_and_room_different_floors(self):
        self.assertEqual(main.extract_cabin_floor("C D46"), "CD")
        self.assertEqual(main.extract_cabin_floor("C46 D"), "CD")


class Test_extract_cabin_room(TestCase):
    def test_none(self):
        self.assertEqual(main.extract_cabin_room(None), [])

    def test_empty(self):
        self.assertEqual(main.extract_cabin_room(""), [])

    def test_single_floor(self):
        self.assertEqual(main.extract_cabin_room("C"), [])

    def test_single_floor_duplicates(self):
        self.assertEqual(main.extract_cabin_room("C C"), [])
        self.assertEqual(main.extract_cabin_room("C C C"), [])

    def test_multiple_floors(self):
        self.assertEqual(main.extract_cabin_room("C D"), [])

    def test_multiple_floors_duplicated(self):
        self.assertEqual(main.extract_cabin_room("C C D"), [])

    def test_single_room(self):
        self.assertEqual(main.extract_cabin_room("C45"), [45])

    def test_multiple_room_same_floor(self):
        self.assertEqual(main.extract_cabin_room("C45 C46"), [45, 46])

    def test_multiple_room_different_floors(self):
        self.assertEqual(main.extract_cabin_room("C45 D46"), [45, 46])

    def test_floor_and_room_same_floor(self):
        self.assertEqual(main.extract_cabin_room("C C45"), [45])
        self.assertEqual(main.extract_cabin_room("C45 C"), [45])

    def test_floor_and_room_different_floors(self):
        self.assertEqual(main.extract_cabin_room("C D46"), [46])
        self.assertEqual(main.extract_cabin_room("C46 D"), [46])


class Test_extract_name_data(TestCase):
    def test_name_1(self):
        self.assertEqual(main.extract_name_data("Abrahim, Mrs. Joseph"),
                         dict(LastName="Abrahim", Salutation="Mrs", Title="Mrs", FirstName="Joseph", SpouseName=None,
                              MaidenName=None))

    def test_name_2(self):
        self.assertEqual(main.extract_name_data("Abrahim, Mrs. Joseph (Sophie Easu)"),
                         dict(LastName="Abrahim", Salutation="Mrs", Title="Mrs", FirstName="Joseph", SpouseName="Sophie",
                              MaidenName="Easu"))

    def test_name_3(self):
        self.assertEqual(main.extract_name_data("Abrahim, Mrs. Joseph (Sophie Halaut Easu)"),
                         dict(LastName="Abrahim", Salutation="Mrs", Title="Mrs", FirstName="Joseph", SpouseName="Sophie Halaut",
                              MaidenName="Easu"))

    def test_name_4(self):
        self.assertEqual(main.extract_name_data("Abrahim, Mrs. (Sophie Halaut Easu)"),
                         dict(LastName="Abrahim", Salutation="Mrs", Title="Mrs", FirstName=None, SpouseName="Sophie Halaut",
                              MaidenName="Easu"))

    def test_name_5(self):
        self.assertEqual(main.extract_name_data("Karnes, Mrs. J Frank (Claire Bennett)"),
                         dict(LastName="Karnes", Salutation="Mrs", Title="Mrs", FirstName="J Frank", SpouseName="Claire",
                              MaidenName="Bennett"))
