import subprocess
import os
import time


def set_wallpaper(image_path):
    # Use PowerShell to set the wallpaper
    command = [
        "powershell",
        "-ExecutionPolicy", "Bypass",
        "-Command", f"$filePath='{image_path}'; $regKey='HKCU:\\Control Panel\\Desktop'; Set-ItemProperty -Path $regKey -Name WallPaper -Value $filePath; rundll32.exe user32.dll,UpdatePerUserSystemParameters"
    ]
    subprocess.run(command, shell=True)


def calc_sleep():
    secdata = time.strftime("%H %M %S").split()
    seconds = (int(secdata[0]) * 3600) + \
        (int(secdata[1]) * 60) + (int(secdata[2]))
    return 86400 + 1 - seconds


if __name__ == "__main__":
    picsdir = r"C:\Users\macko\github\data-science-python\beginner-games\wallpapers"
    images = sorted([os.path.join(picsdir, pic)
                    for pic in os.listdir(picsdir)])

    while True:
        day = int(time.strftime("%w"))
        image = images[day - 1]

        set_wallpaper(image)

        wait = calc_sleep()
        time.sleep(wait)
