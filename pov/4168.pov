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
    sphere { m*<-0.16564364921166563,-0.08377990168809685,-0.4113606726165169>, 1 }        
    sphere {  m*<0.17002133183316603,0.09568485065466983,3.7542865299159014>, 1 }
    sphere {  m*<2.5690647447945913,0.01825407369827728,-1.6405701980677>, 1 }
    sphere {  m*<-1.7872590091045557,2.2446940427305018,-1.3853064380324869>, 1}
    sphere { m*<-1.519471788066724,-2.6429978996733956,-1.1957601528699142>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.17002133183316603,0.09568485065466983,3.7542865299159014>, <-0.16564364921166563,-0.08377990168809685,-0.4113606726165169>, 0.5 }
    cylinder { m*<2.5690647447945913,0.01825407369827728,-1.6405701980677>, <-0.16564364921166563,-0.08377990168809685,-0.4113606726165169>, 0.5}
    cylinder { m*<-1.7872590091045557,2.2446940427305018,-1.3853064380324869>, <-0.16564364921166563,-0.08377990168809685,-0.4113606726165169>, 0.5 }
    cylinder {  m*<-1.519471788066724,-2.6429978996733956,-1.1957601528699142>, <-0.16564364921166563,-0.08377990168809685,-0.4113606726165169>, 0.5}

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
    sphere { m*<-0.16564364921166563,-0.08377990168809685,-0.4113606726165169>, 1 }        
    sphere {  m*<0.17002133183316603,0.09568485065466983,3.7542865299159014>, 1 }
    sphere {  m*<2.5690647447945913,0.01825407369827728,-1.6405701980677>, 1 }
    sphere {  m*<-1.7872590091045557,2.2446940427305018,-1.3853064380324869>, 1}
    sphere { m*<-1.519471788066724,-2.6429978996733956,-1.1957601528699142>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.17002133183316603,0.09568485065466983,3.7542865299159014>, <-0.16564364921166563,-0.08377990168809685,-0.4113606726165169>, 0.5 }
    cylinder { m*<2.5690647447945913,0.01825407369827728,-1.6405701980677>, <-0.16564364921166563,-0.08377990168809685,-0.4113606726165169>, 0.5}
    cylinder { m*<-1.7872590091045557,2.2446940427305018,-1.3853064380324869>, <-0.16564364921166563,-0.08377990168809685,-0.4113606726165169>, 0.5 }
    cylinder {  m*<-1.519471788066724,-2.6429978996733956,-1.1957601528699142>, <-0.16564364921166563,-0.08377990168809685,-0.4113606726165169>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    