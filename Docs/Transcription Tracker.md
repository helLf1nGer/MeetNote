# Transcription Tracker

## Overview

The TranscriptionTracker system provides persistent tracking of transcribed files across sessions. It uses file content hashing to maintain accurate tracking even when files are renamed or moved.

## Key Features

- Content-based file identification using SHA-256 hashing
- Persistent storage in JSON format
- Rename-resistant tracking
- Efficient file status checking
- Integration with GUI file browser

## Technical Implementation

### File Tracking Data Structure 