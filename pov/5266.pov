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
    sphere { m*<-0.5886328085393133,-0.1506496786293727,-1.5421890600630725>, 1 }        
    sphere {  m*<0.3988551613095594,0.28784433254453834,8.39925840041823>, 1 }
    sphere {  m*<3.812193867268056,0.009720451910711109,-3.6192707486525433>, 1 }
    sphere {  m*<-2.226397691619925,2.178019646402337,-2.4882440714685554>, 1}
    sphere { m*<-1.958610470582093,-2.7096722960015605,-2.298697786305985>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3988551613095594,0.28784433254453834,8.39925840041823>, <-0.5886328085393133,-0.1506496786293727,-1.5421890600630725>, 0.5 }
    cylinder { m*<3.812193867268056,0.009720451910711109,-3.6192707486525433>, <-0.5886328085393133,-0.1506496786293727,-1.5421890600630725>, 0.5}
    cylinder { m*<-2.226397691619925,2.178019646402337,-2.4882440714685554>, <-0.5886328085393133,-0.1506496786293727,-1.5421890600630725>, 0.5 }
    cylinder {  m*<-1.958610470582093,-2.7096722960015605,-2.298697786305985>, <-0.5886328085393133,-0.1506496786293727,-1.5421890600630725>, 0.5}

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
    sphere { m*<-0.5886328085393133,-0.1506496786293727,-1.5421890600630725>, 1 }        
    sphere {  m*<0.3988551613095594,0.28784433254453834,8.39925840041823>, 1 }
    sphere {  m*<3.812193867268056,0.009720451910711109,-3.6192707486525433>, 1 }
    sphere {  m*<-2.226397691619925,2.178019646402337,-2.4882440714685554>, 1}
    sphere { m*<-1.958610470582093,-2.7096722960015605,-2.298697786305985>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3988551613095594,0.28784433254453834,8.39925840041823>, <-0.5886328085393133,-0.1506496786293727,-1.5421890600630725>, 0.5 }
    cylinder { m*<3.812193867268056,0.009720451910711109,-3.6192707486525433>, <-0.5886328085393133,-0.1506496786293727,-1.5421890600630725>, 0.5}
    cylinder { m*<-2.226397691619925,2.178019646402337,-2.4882440714685554>, <-0.5886328085393133,-0.1506496786293727,-1.5421890600630725>, 0.5 }
    cylinder {  m*<-1.958610470582093,-2.7096722960015605,-2.298697786305985>, <-0.5886328085393133,-0.1506496786293727,-1.5421890600630725>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    