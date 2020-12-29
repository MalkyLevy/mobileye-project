from part_4.controller import Controller


def main():
    print("--main--")
    my_controller = Controller()
    my_controller.process("./part_4/playlists/dusseldorf_000049.pls")


if __name__ == '__main__':
    main()
