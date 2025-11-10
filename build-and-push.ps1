# Docker Build and Push Script
# Run this to build and push the full Docker image to Docker Hub

param(
    [Parameter(Mandatory=$true)]
    [string]$DockerHubUsername
)

Write-Host "Building full Docker image with all data..." -ForegroundColor Cyan
Write-Host ""

# Build the image
Write-Host "Step 1/4: Building Docker image (this may take 5-10 minutes)..." -ForegroundColor Yellow
docker build -t anticancer-app:full .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Build completed!" -ForegroundColor Green
Write-Host ""

# Tag the image
Write-Host "Step 2/4: Tagging image for Docker Hub..." -ForegroundColor Yellow
docker tag anticancer-app:full "$DockerHubUsername/anticancer-drug-prediction:latest"

Write-Host "Tagged as $DockerHubUsername/anticancer-drug-prediction:latest" -ForegroundColor Green
Write-Host ""

# Login to Docker Hub
Write-Host "Step 3/4: Logging in to Docker Hub..." -ForegroundColor Yellow
docker login

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker Hub login failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Logged in successfully!" -ForegroundColor Green
Write-Host ""

# Push to Docker Hub
Write-Host "Step 4/4: Pushing to Docker Hub (this may take 10-15 minutes for ~2GB)..." -ForegroundColor Yellow
Write-Host "Good time for a coffee break!" -ForegroundColor Cyan
docker push "$DockerHubUsername/anticancer-drug-prediction:latest"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Push failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "SUCCESS! Docker image pushed to Docker Hub!" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Go to Render.com dashboard" -ForegroundColor White
Write-Host "   2. Create New Web Service" -ForegroundColor White
Write-Host "   3. Select 'Deploy an existing image'" -ForegroundColor White
Write-Host "   4. Enter: $DockerHubUsername/anticancer-drug-prediction:latest" -ForegroundColor Yellow
Write-Host "   5. Set Port: 7860" -ForegroundColor White
Write-Host "   6. Deploy!" -ForegroundColor White
Write-Host ""
Write-Host "Or test locally first:" -ForegroundColor Cyan
Write-Host "   docker run -p 7860:7860 $DockerHubUsername/anticancer-drug-prediction:latest" -ForegroundColor Yellow
Write-Host "   Then open: http://localhost:7860" -ForegroundColor White
