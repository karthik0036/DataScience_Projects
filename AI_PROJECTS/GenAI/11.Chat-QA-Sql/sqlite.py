import sqlite3 

#connect to sql
connection = sqlite3.connect("student.db")

#init cursor object to insert ,retreive records
cursor = connection.cursor()

#create table = 
table_info="""
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),
SECTION VARCHAR(25),MARKS INT)
"""

cursor.execute(table_info)
    
#insert records 

cursor.execute(''' INSERT INTO student VALUES('Jeef','Data Science','A',90)''')
cursor.execute(''' INSERT INTO student VALUES('Ned','Data Science','B',56)''')
cursor.execute(''' INSERT INTO student VALUES('John','Data Science','C',78)''')
cursor.execute(''' INSERT INTO student VALUES('Kennedy','Devops','A',80)''')
cursor.execute(''' INSERT INTO student VALUES('Rob','Devops','C',94)''')

## Display all the records
print("The inserted records are")
data=cursor.execute('''Select * from STUDENT''')
for row in data:
    print(row)

## Commit your changes in the database
connection.commit()
connection.close()

