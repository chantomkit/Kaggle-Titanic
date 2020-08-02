import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv", delimiter=",")
print(df.columns.values)

# Fig1

fig1 = plt.figure(figsize=(18,6))

plt.subplot2grid((2,3),(0,0))
df.Survived.value_counts(normalize=True).plot(kind="bar")
plt.title("Survived")

plt.subplot2grid((2,3),(0,1))
df.Embarked.value_counts(normalize=True, sort=False).plot(kind="bar")
plt.title("Embarked from")

plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts(normalize=True, sort=False).plot(kind="bar")
plt.title("Class")

plt.subplot2grid((2,3),(1,0))
plt.scatter(df.Survived, df.Age, alpha=0.1)
plt.title("Age of survivors")

plt.subplot2grid((2,3),(1,1), colspan=2)
for i in [20,40,60,80]:
    df.Survived[(df.Age <= i) & (df.Age > i-20)].plot(kind="kde")
plt.legend(("0-20","21-40","41-60","61-80"))
plt.title("Survivability wrt age")


plt.show()

# Fig2

fig2 = plt.figure(figsize=(18,6))

plt.subplot2grid((2,4),(0,0))
df.Sex.value_counts(normalize=True).plot(kind="bar", rot=0)
plt.title("Gender")

plt.subplot2grid((2,4),(0,1))
for i in ["male", "female"]:
    df.Survived[df.Sex == i].plot(kind="kde")
plt.legend(("male", "female"))
plt.title("Survivability wrt gender")

plt.subplot2grid((2,4),(0,2))
df.Sex[(df.Survived == 1) & (df.Pclass == 1)].value_counts().plot(kind="bar", rot=0)
plt.title("1st class survivors")

plt.subplot2grid((2,4),(0,3))
df.Sex[(df.Survived == 1) & (df.Pclass == 3)].value_counts().plot(kind="bar", rot=0)
plt.title("3rd class survivors")

plt.subplot2grid((2,4),(1,0), colspan=2)
for i in [1,2,3]:
    df.Survived[(df.Sex == "male") & (df.Pclass == i)].plot(kind="kde")
plt.legend(("1st", "2nd", "3rd"))
plt.title("Male survivability wrt class")

plt.subplot2grid((2,4),(1,2), colspan=2)
for i in [1,2,3]:
    df.Survived[(df.Sex == "female") & (df.Pclass == i)].plot(kind="kde")
plt.legend(("1st", "2nd", "3rd"))
plt.title("Female survivability wrt class")

plt.show()