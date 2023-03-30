# Замена пунктуации
import re

#: Замена пунктуации
text2punct = {
    "точка": ".",
    "запятая": ",",
    "кавычки": '"',
    "двоеточие": ":",
    "точка с запятой": ";",
    "восклицательный знак": "!",
    "вопросительный знак": "?",
    "тире": "-",
    "длинное тире": "—",
    "скобки": "()",
    "квадратные скобки": "[]",
    "фигурные скобки": "{}",
}

#: Замена цифр
text2num = {
    "ноль": "0",
    "один": "1",
    "два": "2",
    "три": "3",
    "четыре": "4",
    "пять": "5",
    "шесть": "6",
    "семь": "7",
    "восемь": "8",
    "девять": "9",
    #
    "первый": "1",
    "второй": "2",
    "третий": "3",
    "четвертый": "4",
    "пятый": "5",
    "шестой": "6",
    "седьмой": "7",
    "восьмой": "8",
    "девятый": "9",
    #
    "десять": "10",
    "одиннадцать": "11",
    "двенадцать": "12",
    "тринадцать": "13",
    "четырнадцать": "14",
    "пятнадцать": "15",
    "шестнадцать": "16",
    "семнадцать": "17",
    "восемнадцать": "18",
    "девятнадцать": "19",
    #
    "двадцать": "20",
    "тридцать": "30",
    "сорок": "40",
    "пятьдесят": "50",
    "шестьдесят": "60",
    "семьдесят": "70",
    "восемьдесят": "80",
    "девяносто": "90",
    #
    "сто": "100",
    "двести": "200",
    "триста": "300",
    "четыреста": "400",
    "пятьсот": "500",
    "шестьсот": "600",
    "семьсот": "700",
    "восемьсот": "800",
    "девятьсот": "900",
    #
    "тысяча": "1000",
    "тысячи": "1000",
    "тысяч": "1000",
    #
    "миллион": "1000000",
    "миллиона": "1000000",
    "миллионов": "1000000",
    #
    "миллиард": "1000000000",
    "миллиарда": "1000000000",
    "миллиардов": "1000000000",
}

patterns = {**text2punct, **text2num}
regex = re.compile(r"\b(" + "|".join(patterns.keys()) + r")\b")


def replace_words_with_symbols(text: str) -> str:
    """Замена слов на их символьное обозначение"""

    def replace_match(match):
        return patterns[match.group(0)]

    return regex.sub(replace_match, text)