import pandas as pd
import re


def rename_columns(df):
    # PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

    df = df.rename(columns={
        'Pclass': 'PassengerClass',
        'SibSp': 'SiblingsSpouses',
        'Parch': 'ParentChildren',
    })

    return df


def extract_cabin_floor(cabin):
    if not cabin:
        return None
    else:
        items = cabin.split(" ")
        items = map(lambda item: item[0], items)
        items = list(set(items))
        items.sort()
        return "".join(items)


def extract_cabin_room(cabin):
    if not cabin:
        return []
    else:
        items = cabin.split(" ")
        items = map(lambda item: item[1:], items)
        items = list(set(items))
        items = list(filter(lambda item: item, items))
        items = list(map(lambda item: int(item), items))
        items.sort()
        return items


last_name_regex = re.compile('^(.*?),\s+(.*?)\s+$')

salutation_regex = re.compile('^(.*?).\s+(.*?)$')

salutation_title_map = {
    "Miss": "Miss",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Lady": "Miss",
    "Dona": "Miss",
    "the Countess": "Affluent",
    "Capt": "Affluent",
    "Col": "Affluent",
    "Don": "Affluent",
    "Jonkheer": "Affluent",
    "Major": "Affluent",
    "Mr": "Mr",
    "Dr": "Mr",
    "Master": "Mr",
    "Rev": "Mr",
    "Sir": "Mr",
    "Mrs": "Mrs",
    "Mme": "Mrs"
}


def extract_name_data(name):
    orig_name = name

    name = name.strip()

    last_name = name[:name.index(',')]
    name = name[name.index(',') + 1:]
    name = name.strip()

    salutation = name[:name.index('.')]
    name = name[name.index('.') + 1:]
    name = name.strip()

    paren_position = name.find('(')
    if not paren_position == -1:
        if paren_position == 0:
            first_name = None
        else:
            first_name = name[:paren_position-1]
            name = name[paren_position:]
            name = name.strip()

        name = name[1:len(name) - 1]
        spouse_name = name[:name.rfind(' ')]
        maiden_name = name[name.rfind(' ') + 1:]
    else:
        first_name = name
        spouse_name = None
        maiden_name = None

    return dict(LastName=last_name, Salutation=salutation, Title=salutation_title_map[salutation], FirstName=first_name,
                SpouseName=spouse_name,
                MaidenName=maiden_name)


def convert_name_data_to_series(name):
    name_data = extract_name_data(name)
    return pd.Series([name_data['LastName'], name_data['Salutation'], name_data['Title'], name_data['FirstName'],
                      name_data['SpouseName'], name_data['MaidenName']])


def load_and_clean_input(*files):
    frames = list(map(lambda file: pd.read_csv(file), files))

    for frame in frames:
        if not "Survived" in frame.columns:
            frame['Survived'] = -1

    data = pd.concat(frames)

    data = rename_columns(data)
    data.Cabin = data.Cabin.fillna("")

    data['CabinFloor'] = data.apply(lambda item: extract_cabin_floor(item.Cabin), axis=1)
    data['CabinRooms'] = data.apply(lambda item: extract_cabin_room(item.Cabin), axis=1)
    data['Name'] = data.apply(lambda item: item.Name.replace("\"", ""), axis=1)

    data[['LastName', 'Salutation', 'Title', 'FirstName', 'SpouseName', 'MaidenName']] \
        = data.apply(lambda item: convert_name_data_to_series(item.Name), axis=1)

    data = data[
        ['PassengerId', 'Survived', 'Name', 'Salutation', 'Title', 'FirstName', 'SpouseName', 'MaidenName', 'LastName',
         'Sex', 'Age', 'SiblingsSpouses', 'ParentChildren', 'PassengerClass', 'Embarked', 'Ticket', 'Fare', 'Cabin',
         'CabinFloor', 'CabinRooms']]

    return data


if __name__ == '__main__':
    data = load_and_clean_input("test.csv", "train.csv")
