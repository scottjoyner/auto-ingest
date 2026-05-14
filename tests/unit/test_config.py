#!/usr/bin/env python3
# Test Configuration for Containerized System
# This module provides test-specific settings and utilities

import os
import configparser

class TestConfig:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('/home/scott/git/auto-ingest/tests/test_config.ini')
    
    @property
    def neo4j_test_db(self):
        return self.config.get('neo4j', 'test_db')
    
    @property
    def neo4j_test_password(self):
        return self.config.get('neo4j', 'test_password')
    
    @property
    def neo4j_test_port(self):
        return self.config.get('neo4j', 'test_port')
    
    @property
    def nextcloud_test_url(self):
        return self.config.get('nextcloud', 'test_url')
    
    @property
    def nextcloud_test_user(self):
        return self.config.get('nextcloud', 'test_user')
    
    @property
    def nextcloud_test_password(self):
        return self.config.get('nextcloud', 'test_password')
    
    @property
    def signal_cli_test_url(self):
        return self.config.get('signal_cli', 'test_url')
    
    @property
    def db_test_host(self):
        return self.config.get('db', 'test_host')
    
    @property
    def db_test_port(self):
        return self.config.get('db', 'test_port')

# Global test configuration instance
test_config = TestConfig()
