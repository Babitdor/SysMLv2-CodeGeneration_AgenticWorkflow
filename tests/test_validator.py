import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.sysml_parser.LoggingParsers.SysMLv2LoggingParser import SysMLv2LoggingParser

def test_validator():
    # Test cases
    test_cases = [
        {
            "name": "Valid Package Definition",
            "code": """
                package 'Vehicle' {
                    part def Wheel;
                    part def Chassis;
                    part def Engine;
                }
            """,
            "expected": "valid",
        },
        {
            "name": "Missing Semicolon",
            "code": """
                package 'Vehicle' {
                    part def Wheel
                    part def Chassis;
                }
            """,
            "expected": "invalid",
        },
        {
            "name": "Unclosed Brace",
            "code": """
                package 'Vehicle' {
                    part def Wheel;
                    part def Chassis;
            """,
            "expected": "invalid",
        },
        {
            "name": "Invalid Token",
            "code": """
                package 'Vehicle' {
                    invalid_keyword def Wheel;
                }
            """,
            "expected": "invalid",
        },
        {"name": "Empty Input", "code": "", "expected": "invalid"},
    ]

    print("\nRunning SysML Validator Tests\n" + "=" * 50)

    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print("-" * 50)
        print("Input code:")
        print(test["code"])
        print("\nValidation result:")
        try:
            parser = SysMLv2LoggingParser(test["code"])
            parser.parse()
            result = parser.get_errors()
            print(result)
        except Exception as e:
            print(f"Error during validation: {str(e)}")
        print(f"Expected: {test['expected']}")
        print("=" * 50)


if __name__ == "__main__":
    test_validator()
