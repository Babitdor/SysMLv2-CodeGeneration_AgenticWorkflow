import unittest
from tools.sysml_parser.LoggingParsers.SysMLv2LoggingParser import SysMLv2LoggingParser

class TestSysMLValidator(unittest.TestCase):
    """Test suite for the SysML validator"""

    def test_valid_package_definition(self):
        """Test that a valid package definition is correctly parsed"""
        code = """
            package 'Vehicle' {
                part def Wheel;
                part def Chassis;
                part def Engine;
            }
        """
        parser = SysMLv2LoggingParser(code)
        parser.parse()
        result = parser.get_errors()
        self.assertEqual(result, "Parsing successful!")

    def test_missing_semicolon(self):
        """Test that missing semicolon is detected"""
        code = """
            package 'Vehicle' {
                part def Wheel
                part def Chassis;
            }
        """
        parser = SysMLv2LoggingParser(code)
        parser.parse()
        result = parser.get_errors()
        self.assertTrue("Syntax error" in result)
        self.assertTrue("Unexpected 'part'" in result)

    def test_unclosed_brace(self):
        """Test that unclosed brace is detected"""
        code = """
            package 'Vehicle' {
                part def Wheel;
                part def Chassis;
        """
        parser = SysMLv2LoggingParser(code)
        parser.parse()
        result = parser.get_errors()
        self.assertTrue("Unexpected end of file" in result)
        self.assertTrue("check for unclosed braces" in result)

    def test_invalid_token(self):
        """Test that invalid tokens are detected"""
        code = """
            package 'Vehicle' {
                invalid_keyword def Wheel;
            }
        """
        parser = SysMLv2LoggingParser(code)
        parser.parse()
        result = parser.get_errors()
        self.assertTrue("Syntax error" in result)
        self.assertTrue("Unexpected 'def'" in result)

    def test_empty_input(self):
        """Test that empty input is properly rejected"""
        with self.assertRaises(ValueError) as context:
            SysMLv2LoggingParser("")
        self.assertTrue("Empty input" in str(context.exception))

if __name__ == '__main__':
    unittest.main(verbosity=2)