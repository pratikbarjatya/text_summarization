def demo():
    age = int(input("Enter your age: "))
    year_exp = int(input("Enter your years of experience: "))

    if (16 <= age <= 20) and (year_exp <= 1):
        print("Your Expected Salary: {}".format(10000))
    elif (21 <= age <= 26) and (2 <= year_exp < 5):
        print("Your Expected Salary: {}".format(40000))
    elif (age >= 27) and (year_exp >= 5):
        print("Your Expected Salary: {}".format(90000))
    else:
        print("Not in scoped rules!")


if __name__ == "__main__":
    demo()
