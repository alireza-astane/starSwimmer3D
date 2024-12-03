#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.8357700729023658,0.7271385945214336,0.3600306864273767>, 1 }        
    sphere {  m*<1.07902473569848,0.79034604033312,3.3494815867583503>, 1 }
    sphere {  m*<3.5722719247610164,0.7903460403331197,-0.8678006217322662>, 1 }
    sphere {  m*<-2.307020930891506,5.509837241594387,-1.4981825196744099>, 1}
    sphere { m*<-3.8737954644559136,-7.643080481371389,-2.423898564139982>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.07902473569848,0.79034604033312,3.3494815867583503>, <0.8357700729023658,0.7271385945214336,0.3600306864273767>, 0.5 }
    cylinder { m*<3.5722719247610164,0.7903460403331197,-0.8678006217322662>, <0.8357700729023658,0.7271385945214336,0.3600306864273767>, 0.5}
    cylinder { m*<-2.307020930891506,5.509837241594387,-1.4981825196744099>, <0.8357700729023658,0.7271385945214336,0.3600306864273767>, 0.5 }
    cylinder {  m*<-3.8737954644559136,-7.643080481371389,-2.423898564139982>, <0.8357700729023658,0.7271385945214336,0.3600306864273767>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.8357700729023658,0.7271385945214336,0.3600306864273767>, 1 }        
    sphere {  m*<1.07902473569848,0.79034604033312,3.3494815867583503>, 1 }
    sphere {  m*<3.5722719247610164,0.7903460403331197,-0.8678006217322662>, 1 }
    sphere {  m*<-2.307020930891506,5.509837241594387,-1.4981825196744099>, 1}
    sphere { m*<-3.8737954644559136,-7.643080481371389,-2.423898564139982>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.07902473569848,0.79034604033312,3.3494815867583503>, <0.8357700729023658,0.7271385945214336,0.3600306864273767>, 0.5 }
    cylinder { m*<3.5722719247610164,0.7903460403331197,-0.8678006217322662>, <0.8357700729023658,0.7271385945214336,0.3600306864273767>, 0.5}
    cylinder { m*<-2.307020930891506,5.509837241594387,-1.4981825196744099>, <0.8357700729023658,0.7271385945214336,0.3600306864273767>, 0.5 }
    cylinder {  m*<-3.8737954644559136,-7.643080481371389,-2.423898564139982>, <0.8357700729023658,0.7271385945214336,0.3600306864273767>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    