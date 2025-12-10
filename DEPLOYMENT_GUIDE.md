# 🚀 Step-by-Step Deployment Guide

## ✅ Your Files Are Ready!

All files needed for deployment are prepared:
- ✅ `vix_forecasting_app.py` - Main application
- ✅ `requirements.txt` - Dependencies
- ✅ `VIX 10yr.csv` - Data file
- ✅ `.gitignore` - Git ignore file
- ✅ `README.md` - Documentation

---

## 📦 Step 1: Install Git (if needed)

Check if you have Git:
```bash
git --version
```

If not installed, download from: https://git-scm.com/downloads

---

## 🔧 Step 2: Initialize Git Repository

Open Terminal and run these commands **one by one**:

```bash
# Navigate to your project folder
cd "/Users/sachinshivakumar/Desktop/Econometrics SS/Group Project"

# Initialize git
git init

# Configure git with your info
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add all files
git add vix_forecasting_app.py requirements.txt "VIX 10yr.csv" README.md .gitignore

# Create first commit
git commit -m "Initial commit: VIX Forecasting Dashboard"
```

---

## 🌐 Step 3: Create GitHub Repository

1. Go to https://github.com
2. Click the **"+"** icon (top right) → **"New repository"**
3. Repository name: `vix-forecasting-dashboard`
4. Description: `VIX Volatility Forecasting Dashboard - FIN41660 Project`
5. **Keep it PUBLIC** (required for free Streamlit deployment)
6. **DO NOT** check "Initialize with README" (we already have one)
7. Click **"Create repository"**

---

## 📤 Step 4: Push Code to GitHub

GitHub will show you commands. Copy YOUR repository URL, then run:

```bash
# Add remote repository (REPLACE with YOUR GitHub URL)
git remote add origin https://github.com/YOUR-USERNAME/vix-forecasting-dashboard.git

# Push code
git branch -M main
git push -u origin main
```

**Example:**
```bash
git remote add origin https://github.com/sachin123/vix-forecasting-dashboard.git
git branch -M main
git push -u origin main
```

Enter your GitHub username and password when prompted.

---

## 🎯 Step 5: Deploy to Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit to access your repositories
4. Click **"New app"**
5. Fill in the form:
   - **Repository:** `your-username/vix-forecasting-dashboard`
   - **Branch:** `main`
   - **Main file path:** `vix_forecasting_app.py`
6. Click **"Deploy!"**

---

## ⏱️ Step 6: Wait for Deployment

- Deployment takes **2-5 minutes**
- You'll see logs showing the installation progress
- Once complete, your app will be live! 🎉

---

## 🎊 Step 7: Get Your Public URL

Your app will be available at:
```
https://your-username-vix-forecasting-dashboard.streamlit.app
```

**Share this URL with:**
- Your professor
- Group members
- Include it in your report!

---

## 🔄 Making Updates Later

If you want to update your app:

```bash
cd "/Users/sachinshivakumar/Desktop/Econometrics SS/Group Project"

# Make your changes to the code

# Commit and push
git add .
git commit -m "Description of changes"
git push
```

Streamlit will automatically redeploy! 🚀

---

## ❓ Troubleshooting

### "Git not found"
- Install Git from https://git-scm.com/downloads

### "Permission denied (publickey)"
- Use HTTPS instead of SSH
- URL should start with `https://` not `git@`

### "File not found during deployment"
- Make sure `VIX 10yr.csv` is committed
- Check capitalization matches exactly

### App shows import error
- Check `requirements.txt` has all packages
- Try rebuilding the app from Streamlit dashboard

---

## 📞 Need Help?

- **Streamlit Docs:** https://docs.streamlit.io/streamlit-community-cloud
- **GitHub Docs:** https://docs.github.com/en/get-started

---

**Good luck with your presentation! 🎓**
