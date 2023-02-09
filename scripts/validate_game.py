def validate_solution(grille: list[list[int]]) -> bool:
    """
    Vérifie qu'une grille de Sudoku est bien remplie.

    Table à deux dimensions de 9×9 représentant une grille de Sudoku
    """

    # Lignes
    for i in range(9):
        checksum = 0
        for j in range(9):
            checksum += 2 ** grille[i][j]
        if checksum != 2 ** 10 - 2:
            print(f"La grille n'est pas valide, la ligne {i + 1} n'a pas "
                  f"les bonnes valeurs ({checksum:09b} != {2 ** 10 - 2:09b}).")
            return False

    # Colonnes
    for i in range(9):
        checksum = 0
        for j in range(9):
            checksum += 2 ** grille[j][i]
        if checksum != 2 ** 10 - 2:
            print(f"La grille n'est pas valide, la colonne {i + 1} n'a pas "
                  f"les bonnes valeurs ({checksum:09b} != {2 ** 10 - 2:09b}).")
            return False

    # Secteurs
    for ii in range(3):
        for jj in range(3):
            checksum = 0
            for i in range(3):
                for j in range(3):
                    checksum += 2 ** grille[i + 3 * ii][j + 3 * jj]
            if checksum != 2 ** 10 - 2:
                print(
                    f"La grille n'est pas valide, le secteur ({ii + 1}, {jj + 1}) n'a pas "
                    f"les bonnes valeurs ({checksum:09b} != {2 ** 10 - 2:09b}).")
                return False

    print("La grille est bien résolue.")
    return True
