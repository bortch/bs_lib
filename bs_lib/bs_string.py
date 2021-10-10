import re


def to_snake_case(string):
    if isinstance(string, list):
        results = []
        for s in range(len(string)):
             results.append(to_snake_case(string[s]))
        return results
    else:
        string = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str(string))
        string = re.sub('([a-z0-9])([A-Z])', r'\1_\2', str(string))
        string = re.sub('([\s])+', '_', str(string))
        string = re.sub('(_)+', '_', str(string))
        return string.lower()

def extract_string(from_string,at_position,separator='_'):
    """Extract string from another one

    Args:
        from_string (string): a string with format like {part0}_{part1}_{part2}_{part3}.{extension}
        at_position (int): position of the string to extract
        separator (str, optional): the separator used to split {from_string}. Defaults to '_'.

    Returns:
        string: the extracted string
    """    
    string_list = from_string.split(separator)
    return string_list[at_position]

if __name__ == "__main__":

    print(to_snake_case('abcd_efgh__ijkl___mnop_qrst_uvw_xyz'))
    # get_http_response_code
    print(to_snake_case('abcdEfghIjklMnopQrstUvwXYZ'))
    # http_response_code_xyz
    print(to_snake_case('AbcdEFGH_Ijkl__MnopQrstUvw_XYZ'))
    print(to_snake_case(['abcd_efgh__ijkl___mnop_qrst_uvw_xyz','abcdEfghIjklMnopQrstUvwXYZ','AbcdEFGH_Ijkl__MnopQrstUvw_XYZ']))
