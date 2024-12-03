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
    sphere { m*<0.2512936902676402,0.6829130959541156,0.017547521188250953>, 1 }        
    sphere {  m*<0.4920287950093319,0.8116231741344411,3.0051022923088015>, 1 }
    sphere {  m*<2.986002084273897,0.78494707134049,-1.2116620042629327>, 1 }
    sphere {  m*<-1.37032166962525,3.0113870403727176,-0.9563982442277185>, 1}
    sphere { m*<-3.1463636622122704,-5.739871006322977,-1.9510337077543487>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4920287950093319,0.8116231741344411,3.0051022923088015>, <0.2512936902676402,0.6829130959541156,0.017547521188250953>, 0.5 }
    cylinder { m*<2.986002084273897,0.78494707134049,-1.2116620042629327>, <0.2512936902676402,0.6829130959541156,0.017547521188250953>, 0.5}
    cylinder { m*<-1.37032166962525,3.0113870403727176,-0.9563982442277185>, <0.2512936902676402,0.6829130959541156,0.017547521188250953>, 0.5 }
    cylinder {  m*<-3.1463636622122704,-5.739871006322977,-1.9510337077543487>, <0.2512936902676402,0.6829130959541156,0.017547521188250953>, 0.5}

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
    sphere { m*<0.2512936902676402,0.6829130959541156,0.017547521188250953>, 1 }        
    sphere {  m*<0.4920287950093319,0.8116231741344411,3.0051022923088015>, 1 }
    sphere {  m*<2.986002084273897,0.78494707134049,-1.2116620042629327>, 1 }
    sphere {  m*<-1.37032166962525,3.0113870403727176,-0.9563982442277185>, 1}
    sphere { m*<-3.1463636622122704,-5.739871006322977,-1.9510337077543487>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4920287950093319,0.8116231741344411,3.0051022923088015>, <0.2512936902676402,0.6829130959541156,0.017547521188250953>, 0.5 }
    cylinder { m*<2.986002084273897,0.78494707134049,-1.2116620042629327>, <0.2512936902676402,0.6829130959541156,0.017547521188250953>, 0.5}
    cylinder { m*<-1.37032166962525,3.0113870403727176,-0.9563982442277185>, <0.2512936902676402,0.6829130959541156,0.017547521188250953>, 0.5 }
    cylinder {  m*<-3.1463636622122704,-5.739871006322977,-1.9510337077543487>, <0.2512936902676402,0.6829130959541156,0.017547521188250953>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    