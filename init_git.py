#!/usr/bin/env python3
"""
Script to initialize Git repository for Crop Yield Prediction App
"""
import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def main():
    """Main function to initialize Git repository"""
    print("🚀 Initializing Git repository for Crop Yield Prediction App")
    print("=" * 60)
    
    # Check if Git is installed
    if run_command("git --version", "Checking Git installation") is None:
        print("❌ Git is not installed. Please install Git first.")
        sys.exit(1)
    
    # Check if already a Git repository
    if os.path.exists(".git"):
        print("ℹ️  This directory is already a Git repository")
        choice = input("Do you want to reinitialize? (y/N): ").lower()
        if choice != 'y':
            print("Exiting...")
            sys.exit(0)
        print("🗑️  Removing existing .git directory...")
        run_command("rm -rf .git", "Removing existing Git repository")
    
    # Initialize Git repository
    if run_command("git init", "Initializing Git repository") is None:
        sys.exit(1)
    
    # Add all files
    if run_command("git add .", "Adding all files to Git") is None:
        sys.exit(1)
    
    # Create initial commit
    if run_command('git commit -m "Initial commit: Crop Yield Prediction App"', "Creating initial commit") is None:
        sys.exit(1)
    
    # Create main branch (if not already on main)
    current_branch = run_command("git branch --show-current", "Getting current branch")
    if current_branch and current_branch != "main":
        if run_command("git branch -M main", "Renaming branch to main") is None:
            sys.exit(1)
    
    print("\n🎉 Git repository initialized successfully!")
    print("\n📋 Next steps:")
    print("1. Create a new repository on GitHub")
    print("2. Add the remote origin:")
    print("   git remote add origin https://github.com/MAT-Alpha-375/crop-yield-prediction.git")
    print("3. Push to GitHub:")
    print("   git push -u origin main")
    print("\n🔧 Development commands:")
    print("   make help          - Show available commands")
    print("   make install       - Install dependencies")
    print("   make run           - Run the application")
    print("   make test          - Run tests")
    print("   make lint          - Run linting checks")
    print("   make format        - Format code with black")
    
    # Ask if user wants to set up remote
    setup_remote = input("\n🤔 Do you want to set up a remote repository now? (y/N): ").lower()
    if setup_remote == 'y':
        repo_url = input("Enter the GitHub repository URL: ").strip()
        if repo_url:
            if run_command(f"git remote add origin {repo_url}", "Adding remote origin") is None:
                print("❌ Failed to add remote origin")
            else:
                print("✅ Remote origin added successfully")
                push_choice = input("Do you want to push to GitHub now? (y/N): ").lower()
                if push_choice == 'y':
                    if run_command("git push -u origin main", "Pushing to GitHub") is None:
                        print("❌ Failed to push to GitHub")
                    else:
                        print("🎉 Successfully pushed to GitHub!")
    
    print("\n✨ Setup complete! Happy coding!")

if __name__ == "__main__":
    main()
