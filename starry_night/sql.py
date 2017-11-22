from sqlalchemy import Column, Integer, VARCHAR, DateTime, Float, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import IntegrityError, InvalidRequestError
import logging
import sys


Base = declarative_base()


class SqlEntry(Base):
    '''
    Normal data entry for SQL database
    '''
    __tablename__ = 'global'
    timestamp = Column(DateTime, primary_key=True)
    hashsum = Column(VARCHAR(40))
    sunAlt = Column(Float)
    moonAlt = Column(Float)
    moonPhase = Column(Float)
    brightnessMean = Column(Float)
    brightnessStd = Column(Float)
    global_star_perc = Column(Float)
    global_coverage = Column(Float)


class SqlPoiEntry(Base):
    '''
    Entry for a 'point of interest' object
    '''
    __tablename__ = 'local'
    entry = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(ForeignKey('global.timestamp'))
    ID = Column(Integer, nullable=False)
    ra = Column(Float)
    dec = Column(Float)
    starPerc = Column(Float)


def writeSQL(config, data):
    '''
    Write data to the SQL database specified in config
    '''
    log = logging.getLogger(__name__)
    log.debug('Create SQL engine')
    engine = create_engine(config['SQL']['connection'])
    sMaker = sessionmaker(bind=engine)
    session = scoped_session(sMaker)

    session.add(SqlEntry(
        timestamp=data['timestamp'],
        hashsum=data['hash'],
        sunAlt=data['sun_alt'],
        moonAlt=data['moon_alt'],
        moonPhase=data['moon_phase'],
        brightnessMean=data['brightness_mean'].item(),
        brightnessStd=data['brightness_std'].item(),
        global_star_perc = data['global_star_perc'].item(),
        global_coverage = data['global_coverage'].item()
    ))

    Base.metadata.create_all(engine)
    log.debug('Commit SQL Part 1')
    try:
        session.commit()
    except (IntegrityError, InvalidRequestError) as e:
        log = logging.getLogger(__name__)
        log.error(e)
    except AttributeError as e:
        log.error(e)
        log.error('You might need to hand np.float64.item() to SQL writer because this will return pythons native float object')
        sys.exit(1)


    for i, poi in data['points_of_interest'].iterrows():
        session.add(SqlPoiEntry(timestamp=data['timestamp'], ID=poi.ID, ra=poi.ra, dec=poi.dec, starPerc=poi.starPercentage))
    Base.metadata.create_all(engine)
    log.debug('Commit SQL Part 2')
    try:
        session.commit()
    except (IntegrityError, InvalidRequestError) as e:
        log.error(e)
    except AttributeError as e:
        log.error(e)
        log.error('You might need to hand np.float64.item() to SQL writer because this will return pythons native float object')
        sys.exit(1)
    session.close()
