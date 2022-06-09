# object indexes
scaler_data_index = [
    'product_score',
    'qty',
    'unit_price',
    'freight_price',
    'product_weight_g',
    'lag_price',
    'comp_1',
    'ps1',
    'fp1',
    'comp_2',
    'ps2',
    'fp2',
    'bed_bath_table',
    'computers_accessories',
    'consoles_games',
    'cool_stuff',
    'furniture_decor',
    'garden_tools',
    'health_beauty',
    'perfumery',
    'watches_gifts',
]


def printScalerLogs(scaler_input, scaler_result):
    print('')
    print('Scaler Input')
    print('------------------')
    printScalerDataToConsole(
        scaler_data_index, scaler_input)
    print('Scaler Result')
    print('------------------')
    printScalerDataToConsole(
        scaler_data_index, scaler_result)


def printScalerDataToConsole(scaler_data_index, scaler_data):
    """inputViewer

    Args:
        scaled_result_indexes (array_1_dimension): a one dimension array that contains input indexes
        scaler_data (array_2_dimension): two dimension array that contains the scaled result or input values (the function will call the first value [0] of the first dimension then will iterate to the items insinde that.)
    """
    print("")
    print("---------------------------------------")
    print('Index and The Value Checker')
    print("---------------------------------------")

    i = 0
    for item in scaler_data[0]:
        print(str(scaler_data_index[i]) + ': ' + str(item) + ",")
        i += 1

    print("---------------------------------------")
    print("")
