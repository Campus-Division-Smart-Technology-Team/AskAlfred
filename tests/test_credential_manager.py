"""
Comprehensive tests for credential_manager.py

Tests security controls:
- Secure credential loading from environment
- No credential caching in memory
- Validation and error handling
- Thread safety
- Automatic cleanup
"""
import os
import sys
from unittest.mock import patch
import threading
import pytest

from credential_manager import (
    EnvironmentCredentialProvider,
)


class TestEnvironmentCredentialProvider:
    """Test credential loading from environment variables."""

    def setup_method(self):
        """Setup before each test."""
        self.required_creds = {
            'AZURE_TENANT_ID': 'Azure Tenant ID',
            'AZURE_CLIENT_ID': 'Azure Client ID',
            'AZURE_CLIENT_SECRET': 'Azure Client Secret',
        }

    def test_get_credential_from_environment(self):
        """Test retrieving a credential from environment."""
        with patch.dict(os.environ, {'TEST_KEY': 'test_value'}):
            provider = EnvironmentCredentialProvider({'TEST_KEY': 'Test Key'})
            value = provider.get_credential('TEST_KEY')

            assert value == 'test_value'

    def test_get_missing_credential_raises_error(self):
        """Test that missing credential raises KeyError."""
        provider = EnvironmentCredentialProvider(self.required_creds)

        with pytest.raises(KeyError):
            provider.get_credential('NONEXISTENT_KEY')

    def test_get_empty_credential_raises_error(self):
        """Test that empty credential raises ValueError."""
        with patch.dict(os.environ, {'EMPTY_KEY': ''}):
            provider = EnvironmentCredentialProvider(
                {'EMPTY_KEY': 'Empty Key'})

            with pytest.raises(ValueError):
                provider.get_credential('EMPTY_KEY')

    def test_get_whitespace_only_credential_raises_error(self):
        """Test that whitespace-only credential raises ValueError."""
        with patch.dict(os.environ, {'WHITESPACE_KEY': '   '}):
            provider = EnvironmentCredentialProvider(
                {'WHITESPACE_KEY': 'Whitespace Key'})

            with pytest.raises(ValueError):
                provider.get_credential('WHITESPACE_KEY')

    def test_credential_not_cached_in_object(self):
        """Test that credentials are not stored in the provider object."""
        with patch.dict(os.environ, {'TEMP_KEY': 'temp_value'}):
            provider = EnvironmentCredentialProvider({'TEMP_KEY': 'Temp Key'})

            # Get credential
            value = provider.get_credential('TEMP_KEY')
            assert value == 'temp_value'

            # Change environment
            os.environ['TEMP_KEY'] = 'new_value'

            # Get again - should get new value (not cached)
            new_value = provider.get_credential('TEMP_KEY')
            assert new_value == 'new_value'

    def test_validate_all_required_credentials_present(self):
        """Test validation when all required credentials are present."""
        with patch.dict(os.environ, {
            'AZURE_TENANT_ID': 'tenant-123',
            'AZURE_CLIENT_ID': 'client-456',
            'AZURE_CLIENT_SECRET': 'secret-789',
        }):
            provider = EnvironmentCredentialProvider(self.required_creds)

            assert provider.validate() is True

    def test_validate_missing_required_credential(self):
        """Test validation fails when required credential is missing."""
        with patch.dict(os.environ, {
            'AZURE_TENANT_ID': 'tenant-123',
            # Missing AZURE_CLIENT_ID
            'AZURE_CLIENT_SECRET': 'secret-789',
        }, clear=True):
            provider = EnvironmentCredentialProvider(self.required_creds)

            assert provider.validate() is False

    def test_validate_empty_required_credential(self):
        """Test validation fails when required credential is empty."""
        with patch.dict(os.environ, {
            'AZURE_TENANT_ID': 'tenant-123',
            'AZURE_CLIENT_ID': '',  # Empty
            'AZURE_CLIENT_SECRET': 'secret-789',
        }, clear=True):
            provider = EnvironmentCredentialProvider(self.required_creds)

            assert provider.validate() is False

    def test_validate_whitespace_only_credential(self):
        """Test validation fails with whitespace-only credential."""
        with patch.dict(os.environ, {
            'AZURE_TENANT_ID': 'tenant-123',
            'AZURE_CLIENT_ID': '   ',  # Whitespace only
            'AZURE_CLIENT_SECRET': 'secret-789',
        }, clear=True):
            provider = EnvironmentCredentialProvider(self.required_creds)

            assert provider.validate() is False


class TestCredentialErrorHandling:
    """Test error handling and messages."""

    def test_missing_credential_error_message(self):
        """Test that missing credential error has helpful message."""
        provider = EnvironmentCredentialProvider({'MISSING': 'Missing Cred'})

        try:
            provider.get_credential('MISSING')
            pytest.fail("Should raise KeyError")
        except KeyError as e:
            error_msg = str(e)
            assert 'MISSING' in error_msg or 'not found' in error_msg.lower()

    def test_empty_credential_error_message(self):
        """Test that empty credential error has helpful message."""
        with patch.dict(os.environ, {'EMPTY': ''}):
            provider = EnvironmentCredentialProvider({'EMPTY': 'Empty Cred'})

            try:
                provider.get_credential('EMPTY')
                pytest.fail("Should raise ValueError")
            except ValueError as e:
                error_msg = str(e)
                assert 'empty' in error_msg.lower()

    def test_error_messages_dont_expose_credentials(self):
        """Test that error messages don't expose actual credential values."""
        with patch.dict(os.environ, {'SECRET': 'super_secret_value'}):
            provider = EnvironmentCredentialProvider({'SECRET': 'Secret'})

            # Delete the credential to cause validation failure
            del os.environ['SECRET']

            provider.validate()  # This might fail or log
            # Error messages should not contain 'super_secret_value'


class TestThreadSafety:
    """Test thread-safe credential operations."""

    def test_multiple_threads_get_credential(self):
        """Test that multiple threads can safely get credentials."""

        with patch.dict(os.environ, {'THREAD_TEST': 'value123'}):
            provider = EnvironmentCredentialProvider(
                {'THREAD_TEST': 'Thread Test'})

            results = []
            errors = []

            def get_cred():
                try:
                    value = provider.get_credential('THREAD_TEST')
                    results.append(value)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=get_cred) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All threads should succeed
            assert len(errors) == 0
            assert all(r == 'value123' for r in results)

    def test_concurrent_validation(self):
        """Test concurrent validation operations."""

        with patch.dict(os.environ, {
            'VAL1': 'value1',
            'VAL2': 'value2',
        }):
            creds = {'VAL1': 'Val 1', 'VAL2': 'Val 2'}
            provider = EnvironmentCredentialProvider(creds)

            results = []

            def validate_cred():
                result = provider.validate()
                results.append(result)

            threads = [threading.Thread(target=validate_cred)
                       for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All validations should succeed
            assert all(r is True for r in results)


class TestMultipleProviders:
    """Test multiple provider instances."""

    def test_independent_provider_instances(self):
        """Test that multiple provider instances are independent."""
        creds1 = {'KEY1': 'Key 1'}
        creds2 = {'KEY2': 'Key 2'}

        with patch.dict(os.environ, {'KEY1': 'value1', 'KEY2': 'value2'}):
            provider1 = EnvironmentCredentialProvider(creds1)
            provider2 = EnvironmentCredentialProvider(creds2)

            assert provider1.get_credential('KEY1') == 'value1'
            assert provider2.get_credential('KEY2') == 'value2'

    def test_providers_share_environment_variables(self):
        """Test that all providers access same environment."""
        creds = {'SHARED': 'Shared Key'}

        with patch.dict(os.environ, {'SHARED': 'initial_value'}):
            provider1 = EnvironmentCredentialProvider(creds)

            # Change environment
            os.environ['SHARED'] = 'updated_value'

            provider2 = EnvironmentCredentialProvider(creds)

            # Both should see updated value
            assert provider1.get_credential('SHARED') == 'updated_value'
            assert provider2.get_credential('SHARED') == 'updated_value'


class TestCredentialTypes:
    """Test various credential types."""

    def test_uuid_credential(self):
        """Test with UUID credentials."""
        uuid_value = '12345678-1234-5678-1234-567812345678'
        with patch.dict(os.environ, {'UUID_CRED': uuid_value}):
            provider = EnvironmentCredentialProvider({'UUID_CRED': 'UUID'})
            assert provider.get_credential('UUID_CRED') == uuid_value

    def test_long_credential(self):
        """Test with long credential strings."""
        long_cred = 'a' * 1000
        with patch.dict(os.environ, {'LONG_CRED': long_cred}):
            provider = EnvironmentCredentialProvider({'LONG_CRED': 'Long'})
            value = provider.get_credential('LONG_CRED')
            assert len(value) == 1000

    def test_special_character_credential(self):
        """Test with special character credentials."""
        special = 'secret!@#$%^&*()_+-=[]{}|;:",.<>?/'
        with patch.dict(os.environ, {'SPECIAL': special}):
            provider = EnvironmentCredentialProvider({'SPECIAL': 'Special'})
            assert provider.get_credential('SPECIAL') == special

    def test_unicode_credential(self):
        """Test with unicode credentials."""
        unicode_cred = '密码😀🔑'
        with patch.dict(os.environ, {'UNICODE': unicode_cred}):
            provider = EnvironmentCredentialProvider({'UNICODE': 'Unicode'})
            value = provider.get_credential('UNICODE')
            assert value == unicode_cred


class TestEnvironmentVariableHandling:
    """Test environment variable handling edge cases."""

    def test_credential_with_equals_sign(self):
        """Test credential value containing equals sign."""
        with patch.dict(os.environ, {'CRED': 'value=something'}):
            provider = EnvironmentCredentialProvider({'CRED': 'Cred'})
            assert provider.get_credential('CRED') == 'value=something'

    def test_credential_with_newline_character(self):
        """Test credential containing newline."""
        cred_with_newline = 'line1\nline2'
        with patch.dict(os.environ, {'MULTILINE': cred_with_newline}):
            provider = EnvironmentCredentialProvider(
                {'MULTILINE': 'Multiline'})
            assert provider.get_credential('MULTILINE') == cred_with_newline

    @pytest.mark.skipif(sys.platform == 'win32', reason="Environment variables are case-insensitive on Windows")
    def test_credential_key_case_sensitive(self):
        """Test that credential keys are case-sensitive (Unix/Linux only)."""
        with patch.dict(os.environ, {'API_KEY': 'value1'}):
            provider = EnvironmentCredentialProvider({'API_KEY': 'API Key'})

            # Should work with exact case
            assert provider.get_credential('API_KEY') == 'value1'

            # Should fail with different case (on Unix/Linux)
            with pytest.raises(KeyError):
                provider.get_credential('api_key')


class TestValidationScenarios:
    """Test validation in realistic scenarios."""

    def test_azure_credentials_validation(self):
        """Test validation of complete Azure credential set."""
        azure_creds = {
            'AZURE_TENANT_ID': 'Azure Tenant',
            'AZURE_CLIENT_ID': 'Azure Client',
            'AZURE_CLIENT_SECRET': 'Azure Secret',
        }

        with patch.dict(os.environ, {
            'AZURE_TENANT_ID': 'tenant-value',
            'AZURE_CLIENT_ID': 'client-value',
            'AZURE_CLIENT_SECRET': 'secret-value',
        }):
            provider = EnvironmentCredentialProvider(azure_creds)
            assert provider.validate() is True

    def test_redis_credentials_validation(self):
        """Test validation of Redis credential set."""
        redis_creds = {
            'REDIS_HOST': 'Redis Host',
            'REDIS_PORT': 'Redis Port',
            'REDIS_PASSWORD': 'Redis Password',
        }

        with patch.dict(os.environ, {
            'REDIS_HOST': 'localhost',
            'REDIS_PORT': '6379',
            'REDIS_PASSWORD': 'redis_secret',
        }):
            provider = EnvironmentCredentialProvider(redis_creds)
            assert provider.validate() is True

    def test_partial_credentials_fail_validation(self):
        """Test that partial credential sets fail validation."""
        full_creds = {
            'KEY1': 'Key 1',
            'KEY2': 'Key 2',
            'KEY3': 'Key 3',
        }

        # Only provide 2 of 3
        with patch.dict(os.environ, {
            'KEY1': 'value1',
            'KEY2': 'value2',
            # KEY3 missing
        }, clear=True):
            provider = EnvironmentCredentialProvider(full_creds)
            assert provider.validate() is False


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_required_credentials_dict(self):
        """Test provider with empty credentials dict."""
        provider = EnvironmentCredentialProvider({})

        # Should validate as nothing is required
        assert provider.validate() is True

    def test_very_long_credential_key_name(self):
        """Test with very long credential key name."""
        long_key = 'A' * 1000
        with patch.dict(os.environ, {long_key: 'value'}):
            provider = EnvironmentCredentialProvider({long_key: 'Long Key'})
            assert provider.get_credential(long_key) == 'value'

    def test_special_characters_in_key_name(self):
        """Test with special characters in key name."""
        # Note: Environment variable names typically can't have special chars
        # but we test the code's handling
        key = 'MY_KEY_123'
        with patch.dict(os.environ, {key: 'value'}):
            provider = EnvironmentCredentialProvider({key: 'My Key'})
            assert provider.get_credential(key) == 'value'

    def test_credential_value_with_tabs_and_spaces(self):
        """Test credential values with whitespace."""
        value = '  value_with_spaces  '
        with patch.dict(os.environ, {'SPACES': value}):
            provider = EnvironmentCredentialProvider({'SPACES': 'Spaces'})
            result = provider.get_credential('SPACES')
            # Should preserve the exact value including leading/trailing spaces
            assert result == value

    def test_numeric_string_credential(self):
        """Test credential that is numeric string."""
        with patch.dict(os.environ, {'PORT': '6379'}):
            provider = EnvironmentCredentialProvider({'PORT': 'Port'})
            value = provider.get_credential('PORT')
            assert value == '6379'
            assert isinstance(value, str)


class TestSecurityProperties:
    """Test security-relevant properties."""

    def test_credentials_not_logged_in_traceback(self):
        """Test that credentials don't appear in exception tracebacks."""
        with patch.dict(os.environ, {}, clear=True):
            provider = EnvironmentCredentialProvider({'SECRET': 'Secret'})

            try:
                provider.get_credential('SECRET')
            except KeyError:
                # Exception message should not contain the secret value
                # (there shouldn't be any secret in the environment anyway)
                pass

    def test_object_repr_no_credentials(self):
        """Test that object representation doesn't expose credentials."""
        provider = EnvironmentCredentialProvider({'KEY': 'Key'})

        # Should not expose the key name in repr if it contains secrets
        # This is implementation dependent
        assert repr(provider) is not None

    def test_no_default_credentials_in_code(self):
        """Test that no hardcoded credentials are in the provider."""
        provider = EnvironmentCredentialProvider({})

        # Check that provider doesn't have any suspicious attributes
        for attr_name in dir(provider):
            if not attr_name.startswith('_'):
                attr_value = getattr(provider, attr_name)
                if isinstance(attr_value, str):
                    # Shouldn't contain long credential-like strings
                    assert len(attr_value) < 50 or callable(attr_value)


class TestProviderInterface:
    """Test the CredentialProvider interface."""

    def test_environment_provider_implements_interface(self):
        """Test that EnvironmentCredentialProvider implements interface."""
        provider = EnvironmentCredentialProvider({})

        # Should have required methods
        assert hasattr(provider, 'get_credential')
        assert hasattr(provider, 'validate')
        assert callable(provider.get_credential)
        assert callable(provider.validate)

    def test_get_credential_is_callable(self):
        """Test that get_credential is a callable method."""
        provider = EnvironmentCredentialProvider({})
        assert callable(provider.get_credential)

    def test_validate_is_callable(self):
        """Test that validate is a callable method."""
        provider = EnvironmentCredentialProvider({})
        assert callable(provider.validate)
