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
    sphere { m*<-0.23818261834726456,-0.1225631792602697,-1.3115789584612791>, 1 }        
    sphere {  m*<0.44069037702430514,0.24039924120824713,7.113325496098668>, 1 }
    sphere {  m*<2.4965257756589927,-0.020529203873895452,-2.540788483912459>, 1 }
    sphere {  m*<-1.8597979782401544,2.20591076515833,-2.285524723877246>, 1}
    sphere { m*<-1.5920107572023225,-2.6817811772455675,-2.095978438714673>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.44069037702430514,0.24039924120824713,7.113325496098668>, <-0.23818261834726456,-0.1225631792602697,-1.3115789584612791>, 0.5 }
    cylinder { m*<2.4965257756589927,-0.020529203873895452,-2.540788483912459>, <-0.23818261834726456,-0.1225631792602697,-1.3115789584612791>, 0.5}
    cylinder { m*<-1.8597979782401544,2.20591076515833,-2.285524723877246>, <-0.23818261834726456,-0.1225631792602697,-1.3115789584612791>, 0.5 }
    cylinder {  m*<-1.5920107572023225,-2.6817811772455675,-2.095978438714673>, <-0.23818261834726456,-0.1225631792602697,-1.3115789584612791>, 0.5}

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
    sphere { m*<-0.23818261834726456,-0.1225631792602697,-1.3115789584612791>, 1 }        
    sphere {  m*<0.44069037702430514,0.24039924120824713,7.113325496098668>, 1 }
    sphere {  m*<2.4965257756589927,-0.020529203873895452,-2.540788483912459>, 1 }
    sphere {  m*<-1.8597979782401544,2.20591076515833,-2.285524723877246>, 1}
    sphere { m*<-1.5920107572023225,-2.6817811772455675,-2.095978438714673>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.44069037702430514,0.24039924120824713,7.113325496098668>, <-0.23818261834726456,-0.1225631792602697,-1.3115789584612791>, 0.5 }
    cylinder { m*<2.4965257756589927,-0.020529203873895452,-2.540788483912459>, <-0.23818261834726456,-0.1225631792602697,-1.3115789584612791>, 0.5}
    cylinder { m*<-1.8597979782401544,2.20591076515833,-2.285524723877246>, <-0.23818261834726456,-0.1225631792602697,-1.3115789584612791>, 0.5 }
    cylinder {  m*<-1.5920107572023225,-2.6817811772455675,-2.095978438714673>, <-0.23818261834726456,-0.1225631792602697,-1.3115789584612791>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    