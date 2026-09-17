from skypy.galaxies.redshift import redshifts_from_comoving_density
from slsim.Sources.Events.event_pop import EventPopulation
from astropy import units
import numpy as np


class EventLightcone(object):
    """Class to sample observer-frame events within a sky area as a function of
    redshift."""

    def __init__(self, cosmo, redshifts, sky_area, noise, time_interval, model):
        """
        :param cosmo: cosmology object
        :type cosmo: ~astropy.cosmology object
        :param redshifts: redshifts for event density lightcone to be evaluated at.
            Must be sorted in ascending order.
        :type redshifts: array-like
        :param sky_area: sky area for sampled event in [solid angle]
        :type sky_area: `~Astropy.units.Quantity`
        :param noise: poisson-sample the number of event in the event density lightcone
        :type noise: bool
        :param time_interval: time interval for event density lightcone to be evaluated over
        :type time_interval: `~Astropy.units.Quantity`
        :param model : name of model, chosen form "BNS" or "SNIa"
        :type model: string value
        """
        self._cosmo = cosmo
        self._input_redshifts = np.asarray(redshifts)
        self._sky_area = sky_area
        self._noise = noise
        self._time_interval = time_interval
        self._model = model

        # check if redshift array is sorted in ascending order
        if np.any(np.diff(self._input_redshifts) <= 0):
            raise ValueError("redshifts must be sorted in strictly increasing order.")

        event_pop = EventPopulation(
            model=self._model, cosmo=self._cosmo, z_max=self._input_redshifts[-1]
        )

        # Convert source-frame event rate to observer-frame event rate
        rate_source_frame = event_pop.event_rate(self._input_redshifts)
        rate_observer_frame = rate_source_frame / (1 + self._input_redshifts)

        self._density = self.convert_density(rate_observer_frame)

    def convert_density(self, density):
        """Converts event rate densities from [yr^(-1)Mpc^(-3)] to event
        density over the time interval.

        :param density: initial event rate density, such as BNS merger
            or SNIa, in unit [yr^(-1)Mpc^(-3)]
        :return: event rate density in [Mpc^(-3)].
        :return type: array-like
        """
        time_conversion_in_years = self._time_interval.to_value(units.year)
        converted_density = density * time_conversion_in_years

        return converted_density

    def event_sample(self):
        """Samples event redshifts in the light cone.

        The sampled events correspond to the input sky area and input
        observer-frame time interval.

        :return: sampled event redshifts. The length of the array is the
            number of sampled events over the input time interval.
            Redshift is dimensionless.
        :return type: numpy.ndarray
        """
        if not hasattr(self, "_output_redshifts"):
            self._output_redshifts = redshifts_from_comoving_density(
                redshift=self._input_redshifts,
                density=self._density,
                sky_area=self._sky_area,
                cosmology=self._cosmo,
                noise=self._noise,
            )
        return self._output_redshifts
