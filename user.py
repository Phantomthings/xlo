import bcrypt
from getpass import getpass
from sqlalchemy import create_engine, text

DATABASE_URL_ELTO = "mysql+pymysql://AdminNidec:u6Ehe987XBSXxa4@141.94.31.144:3306/indicator"
DATABASE_URL_IE = "mysql+pymysql://nidec:MaV38f5xsGQp83@162.19.251.55:3306/Charges"

engine_elto = create_engine(DATABASE_URL_ELTO)
engine_ie = create_engine(DATABASE_URL_IE)


def hash_password(password: str) -> str:
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def user_exists(engine, username: str) -> bool:
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) as count FROM users WHERE username = :username"),
            {"username": username}
        )
        count = result.fetchone()[0]
        return count > 0


def create_user_in_db(engine, db_name: str, username: str, password_hash: str) -> bool:
    if user_exists(engine, username):
        print(f"⚠️  L'utilisateur '{username}' existe déjà dans {db_name}.")
        return False
    
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO users (username, password_hash, is_active)
                    VALUES (:username, :password_hash, :is_active)
                """),
                {
                    "username": username,
                    "password_hash": password_hash,
                    "is_active": True
                }
            )
            conn.commit()
        print(f"✅ Utilisateur '{username}' créé dans {db_name}!")
        return True
    except Exception as e:
        print(f"❌ Erreur dans {db_name}: {e}")
        return False


def create_user_both_dbs(username: str, password: str):
    password_hash = hash_password(password)
    
    print("\n📦 Création dans les deux bases de données...")
    
    success_elto = create_user_in_db(engine_elto, "ELTO (indicator)", username, password_hash)
    success_ie = create_user_in_db(engine_ie, "IE Charge (Charges)", username, password_hash)
    
    print()
    if success_elto or success_ie:
        print("✅ Utilisateur créé avec succès!")
    else:
        print("⚠️  L'utilisateur existe déjà dans les deux bases.")
    
    engine_elto.dispose()
    engine_ie.dispose()


def main():
    print("=" * 50)
    print("🔐 Création d'un utilisateur (ELTO + IE Charge)")
    print("=" * 50)
    print()
    
    username = input("Nom d'utilisateur : ").strip()
    if not username:
        print("❌ Le nom d'utilisateur ne peut pas être vide.")
        return
    
    password = getpass("Mot de passe : ")
    if len(password) < 6:
        print("❌ Le mot de passe doit faire au moins 6 caractères.")
        return
    
    password_confirm = getpass("Confirmer le mot de passe : ")
    if password != password_confirm:
        print("❌ Les mots de passe ne correspondent pas.")
        return
    
    create_user_both_dbs(username, password)


if __name__ == "__main__":
    main()