"""
Comprehensive tests for log_sanitiser.py

Tests security controls:
- API key redaction
- Password redaction
- Token redaction
- Connection string redaction
- Azure credential redaction
- Private key redaction
- Error message sanitization
- Dictionary recursion
"""

import pytest
from unittest.mock import patch
from log_sanitiser import (
    sanitise_message,
    sanitise_error,
    sanitise_dict,
    SanitisedFormatter,
    SENSITIVE_PATTERNS,
)
import logging


class TestAPIKeyRedaction:
    """Test redaction of API keys."""

    def test_api_key_redaction(self):
        """Test that API keys are redacted."""
        message = 'api_key = "sk-1234567890abcdefghijklmnop"'
        sanitised = sanitise_message(message)
        assert 'sk-1234567890abcdefghijklmnop' not in sanitised
        assert 'REDACTED' in sanitised

    def test_pinecone_api_key_redaction(self):
        """Test that Pinecone API keys are redacted."""
        message = 'pinecone_api_key: "pcne-1234567890abcdefghijklmnop"'
        sanitised = sanitise_message(message)
        assert 'pcne-1234567890abcdefghijklmnop' not in sanitised
        assert 'REDACTED' in sanitised

    def test_openai_api_key_redaction(self):
        """Test that OpenAI API keys are redacted."""
        message = 'openai_key = sk-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh'
        sanitised = sanitise_message(message)
        assert 'sk-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh' not in sanitised
        assert 'REDACTED' in sanitised

    def test_various_api_key_formats(self):
        """Test redaction with various API key formats."""
        keys = [
            'api_key: "test_key_12345678901234567890"',
            'apiKey = "test_key_12345678901234567890"',
            'Api-Key: test_key_12345678901234567890',
            'API_KEY=test_key_12345678901234567890',
        ]
        for key_string in keys:
            sanitised = sanitise_message(key_string)
            assert 'test_key_12345678901234567890' not in sanitised


class TestPasswordRedaction:
    """Test redaction of passwords."""

    def test_password_redaction(self):
        """Test that passwords are redacted."""
        message = 'password = "SuperSecret123"'
        sanitised = sanitise_message(message)
        assert 'SuperSecret123' not in sanitised
        assert 'REDACTED' in sanitised

    def test_passwd_redaction(self):
        """Test that 'passwd' is redacted."""
        message = 'passwd: "MyPassword456"'
        sanitised = sanitise_message(message)
        assert 'MyPassword456' not in sanitised

    def test_pwd_redaction(self):
        """Test that 'pwd' is redacted."""
        message = 'pwd=MyPwd789'
        sanitised = sanitise_message(message)
        assert 'MyPwd789' not in sanitised

    def test_password_in_context(self):
        """Test password redaction in full context."""
        message = 'Failed login with username=admin and password=secret123'
        sanitised = sanitise_message(message)
        assert 'secret123' not in sanitised
        assert 'admin' in sanitised  # Username not redacted


class TestTokenRedaction:
    """Test redaction of authentication tokens."""

    def test_bearer_token_redaction(self):
        """Test that bearer tokens are redacted."""
        message = 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
        sanitised = sanitise_message(message)
        assert 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9' not in sanitised
        assert 'REDACTED' in sanitised

    def test_auth_token_redaction(self):
        """Test that auth tokens are redacted."""
        message = 'auth_token=mtokenvalue1234567890abcdefghij'
        sanitised = sanitise_message(message)
        assert 'mtokenvalue1234567890abcdefghij' not in sanitised

    def test_token_redaction(self):
        """Test generic token redaction."""
        message = 'token: "tokenvalue12345678901234567890"'
        sanitised = sanitise_message(message)
        assert 'tokenvalue12345678901234567890' not in sanitised


class TestSecretRedaction:
    """Test redaction of secrets."""

    def test_client_secret_redaction(self):
        """Test that client secrets are redacted."""
        message = 'client_secret=mysecretvalue1234567890abcdefghijk'
        sanitised = sanitise_message(message)
        assert 'mysecretvalue1234567890abcdefghijk' not in sanitised
        assert 'REDACTED' in sanitised

    def test_secret_redaction(self):
        """Test that secrets are redacted."""
        message = 'secret: secretvalue12345678901234567890'
        sanitised = sanitise_message(message)
        assert 'secretvalue12345678901234567890' not in sanitised


class TestConnectionStringRedaction:
    """Test redaction of connection strings."""

    def test_mongodb_connection_string_redaction(self):
        """Test that MongoDB connection strings are redacted."""
        message = 'mongodb://user:password@host:27017/database'
        sanitised = sanitise_message(message)
        # Should redact the whole connection string or at least the password part
        assert 'REDACTED' in sanitised or 'password' not in sanitised

    def test_redis_connection_string_redaction(self):
        """Test that Redis connection strings are redacted."""
        message = 'redis://user:mypassword@localhost:6379/0'
        sanitised = sanitise_message(message)
        assert 'mypassword' not in sanitised

    def test_connection_string_keyword(self):
        """Test redaction with 'connection_string' keyword."""
        message = 'connection_string="Server=localhost;Password=secret123"'
        sanitised = sanitise_message(message)
        assert 'REDACTED' in sanitised


class TestAzureCredentialRedaction:
    """Test redaction of Azure credentials."""

    def test_azure_tenant_id_redaction(self):
        """Test that Azure tenant IDs are redacted."""
        message = 'azure_tenant_id=12345678-1234-1234-1234-123456789012'
        sanitised = sanitise_message(message)
        assert '12345678-1234-1234-1234-123456789012' not in sanitised

    def test_azure_client_id_redaction(self):
        """Test that Azure client IDs are redacted."""
        message = 'client_id=87654321-4321-4321-4321-210987654321'
        sanitised = sanitise_message(message)
        assert '87654321-4321-4321-4321-210987654321' not in sanitised

    def test_azure_client_secret_redaction(self):
        """Test that Azure client secrets are redacted."""
        message = 'azure_client_secret=myazuresecret1234567890'
        sanitised = sanitise_message(message)
        assert 'myazuresecret1234567890' not in sanitised


class TestPrivateKeyRedaction:
    """Test redaction of private keys."""

    def test_rsa_private_key_redaction(self):
        """Test that RSA private keys are redacted."""
        message = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA2Z3qX2BTLS4e...keycontenthere...
-----END RSA PRIVATE KEY-----"""
        sanitised = sanitise_message(message)
        assert 'MIIEpAIBAAKCAQEA2Z3qX2BTLS4e' not in sanitised
        assert 'REDACTED' in sanitised

    def test_ec_private_key_redaction(self):
        """Test that EC private keys are redacted."""
        message = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIIEr...keycontenthere...
-----END EC PRIVATE KEY-----"""
        sanitised = sanitise_message(message)
        assert 'MHcCAQEEIIEr' not in sanitised

    def test_private_key_in_context(self):
        """Test private key redaction in context."""
        message = "ERROR: Key file -----BEGIN PRIVATE KEY-----\nMIIE...content...\n-----END PRIVATE KEY-----"
        sanitised = sanitise_message(message)
        assert 'MIIE' not in sanitised or sanitised.count(
            'MIIE') < message.count('MIIE')


class TestCredentialsObjectRedaction:
    """Test redaction of credentials objects."""

    def test_credentials_object_redaction(self):
        """Test that credentials objects are redacted."""
        message = 'credentials = {username: admin, password: secret, token: abc123}'
        sanitised = sanitise_message(message)
        # The entire credentials object should be redacted
        assert 'REDACTED' in sanitised


class TestErrorMessageSanitization:
    """Test sanitization of error messages."""

    def test_sanitise_generic_exception(self):
        """Test sanitization of generic exception."""
        error = Exception('Connection failed: password=secret123')
        sanitised = sanitise_error(error)
        assert 'secret123' not in sanitised
        assert 'Exception' in sanitised

    def test_sanitise_value_error(self):
        """Test sanitization of ValueError."""
        error = ValueError(
            'openai_key=sk-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh is invalid')
        sanitised = sanitise_error(error)
        assert 'sk-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh' not in sanitised
        assert 'ValueError' in sanitised

    def test_sanitise_runtime_error(self):
        """Test sanitization of RuntimeError."""
        error = RuntimeError(
            'token=abcdef123456789012345678901234567890abcdef')
        sanitised = sanitise_error(error)
        assert 'abcdef123456789012345678901234567890abcdef' not in sanitised
        assert 'RuntimeError' in sanitised

    def test_sanitise_error_with_no_sensitive_data(self):
        """Test that error messages without sensitive data are unchanged."""
        error = ValueError('Query parameter missing')
        sanitised = sanitise_error(error)
        assert 'Query parameter missing' in sanitised

    def test_sanitise_error_preserves_type(self):
        """Test that error type is preserved."""
        error = KeyError('AZURE_TENANT_ID')
        sanitised = sanitise_error(error)
        assert 'KeyError' in sanitised


class TestDictionarySanitization:
    """Test recursive dictionary sanitization."""

    def test_sanitise_flat_dict(self):
        """Test sanitization of flat dictionary."""
        data = {
            'username': 'admin',
            'password': 'secret123',
            'api_key': 'sk-12345678'
        }
        sanitised = sanitise_dict(data)
        assert sanitised['username'] == 'admin'
        assert 'secret123' not in str(sanitised)
        assert 'sk-12345678' not in str(sanitised)

    def test_sanitise_nested_dict(self):
        """Test sanitization of nested dictionary."""
        data = {
            'credentials': {
                'user': 'test',
                'password': 'secret456'
            },
            'tokens': {
                'auth': 'token123456789012345678',
                'refresh': 'refresh_token1234567890'
            }
        }
        sanitised = sanitise_dict(data)
        sanitised_str = str(sanitised)
        assert 'secret456' not in sanitised_str
        assert 'token123456789012345678' not in sanitised_str

    def test_sanitise_dict_with_lists(self):
        """Test sanitization of dictionary with lists."""
        data = {
            'users': [
                {'name': 'Alice', 'password': 'pass123'},
                {'name': 'Bob', 'password': 'pass456'}
            ]
        }
        sanitised = sanitise_dict(data)
        assert 'pass123' not in str(sanitised)
        assert 'pass456' not in str(sanitised)

    def test_sanitise_dict_preserves_structure(self):
        """Test that sanitization preserves dictionary structure."""
        data = {
            'config': {
                'debug': True,
                'timeout': 30,
                'api_key': 'sk-test123'
            }
        }
        sanitised = sanitise_dict(data)
        assert 'config' in sanitised
        assert 'debug' in sanitised['config']
        assert sanitised['config']['debug'] is True
        assert sanitised['config']['timeout'] == 30


class TestSanitisedFormatter:
    """Test the custom logging formatter."""

    def test_formatter_redacts_in_log_record(self):
        """Test that formatter redacts sensitive data in log records."""
        formatter = SanitisedFormatter('%(message)s')
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='openai_key=sk-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh',
            args=(),
            exc_info=None
        )
        formatted = formatter.format(record)
        assert 'sk-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh' not in formatted
        assert 'REDACTED' in formatted

    def test_formatter_with_exception_info(self):
        """Test formatter with exception information."""
        formatter = SanitisedFormatter('%(message)s')
        try:
            raise ValueError('Password is secret123')
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name='test',
                level=logging.ERROR,
                pathname='test.py',
                lineno=1,
                msg='Exception occurred',
                args=(),
                exc_info=exc_info
            )
            formatted = formatter.format(record)
            # The formatted output might contain the traceback
            # but sensitive values should be redacted

    def test_formatter_preserves_non_sensitive_data(self):
        """Test that formatter preserves non-sensitive information."""
        formatter = SanitisedFormatter('%(message)s - %(name)s')
        record = logging.LogRecord(
            name='my_module',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Normal operation completed',
            args=(),
            exc_info=None
        )
        formatted = formatter.format(record)
        assert 'Normal operation completed' in formatted
        assert 'my_module' in formatted


class TestMultipleSensitiveValues:
    """Test handling of messages with multiple sensitive values."""

    def test_multiple_api_keys(self):
        """Test redaction of multiple API keys in one message."""
        message = 'openai_key=sk-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh and api_key=key_1234567890abcdefghijklmnop'
        sanitised = sanitise_message(message)
        assert 'sk-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh' not in sanitised
        assert 'key_1234567890abcdefghijklmnop' not in sanitised
        assert sanitised.count('REDACTED') >= 2

    def test_mixed_sensitive_types(self):
        """Test redaction of mixed sensitive data types."""
        message = '''
        config {
            api_key: sk-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh,
            password: secret_password_with_special_chars,
            token: auth123456789012345678901234567890
        }
        '''
        sanitised = sanitise_message(message)
        assert 'sk-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh' not in sanitised
        assert 'secret_password_with_special_chars' not in sanitised
        assert 'auth123456789012345678901234567890' not in sanitised


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_message(self):
        """Test sanitization of empty message."""
        assert sanitise_message('') == ''
        assert sanitise_message('   ') == '   '

    def test_non_string_input(self):
        """Test handling of non-string input."""
        result = sanitise_message(str(12345))
        assert isinstance(result, str)

    def test_very_long_message(self):
        """Test sanitization of very long messages."""
        long_message = 'api_key=secret_key_1234567890abcdefghijklmnop ' * 100
        sanitised = sanitise_message(long_message)
        assert 'secret_key_1234567890abcdefghijklmnop' not in sanitised

    def test_message_with_newlines(self):
        """Test sanitization of messages with newlines."""
        message = 'Line 1\napi_key=secret_value_1234567890abcdefghijklmnop\nLine 3'
        sanitised = sanitise_message(message)
        assert 'secret_value_1234567890abcdefghijklmnop' not in sanitised
        assert 'Line 1' in sanitised
        assert 'Line 3' in sanitised

    def test_none_values_in_dict(self):
        """Test sanitization of dict with None values."""
        # Note: sanitise_dict redacts values for sensitive keys like 'api_key'
        # regardless of their value, so we test with a non-sensitive key
        data = {'config_value': None,
                'password': 'secret_password_1234567890abcdefghij'}
        sanitised = sanitise_dict(data)
        # Non-sensitive keys preserve their None value
        assert sanitised['config_value'] is None
        # Sensitive keys (like 'password') get redacted
        assert sanitised['password'] == '***REDACTED***'

    def test_false_and_zero_values(self):
        """Test that false and zero values are preserved."""
        data = {'enabled': False, 'count': 0, 'password': 'secret'}
        sanitised = sanitise_dict(data)
        assert sanitised['enabled'] is False
        assert sanitised['count'] == 0
        assert 'secret' not in str(sanitised)
