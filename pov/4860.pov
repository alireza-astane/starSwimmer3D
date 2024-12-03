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
    sphere { m*<-0.24882510299756053,-0.1282532302930939,-1.443653612997787>, 1 }        
    sphere {  m*<0.4755468271701885,0.25903539495150424,7.54589869330394>, 1 }
    sphere {  m*<2.4858832910086965,-0.02621925490671974,-2.6728631384489687>, 1 }
    sphere {  m*<-1.8704404628904505,2.2002207141255052,-2.4175993784137555>, 1}
    sphere { m*<-1.6026532418526187,-2.687471228278392,-2.228053093251183>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4755468271701885,0.25903539495150424,7.54589869330394>, <-0.24882510299756053,-0.1282532302930939,-1.443653612997787>, 0.5 }
    cylinder { m*<2.4858832910086965,-0.02621925490671974,-2.6728631384489687>, <-0.24882510299756053,-0.1282532302930939,-1.443653612997787>, 0.5}
    cylinder { m*<-1.8704404628904505,2.2002207141255052,-2.4175993784137555>, <-0.24882510299756053,-0.1282532302930939,-1.443653612997787>, 0.5 }
    cylinder {  m*<-1.6026532418526187,-2.687471228278392,-2.228053093251183>, <-0.24882510299756053,-0.1282532302930939,-1.443653612997787>, 0.5}

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
    sphere { m*<-0.24882510299756053,-0.1282532302930939,-1.443653612997787>, 1 }        
    sphere {  m*<0.4755468271701885,0.25903539495150424,7.54589869330394>, 1 }
    sphere {  m*<2.4858832910086965,-0.02621925490671974,-2.6728631384489687>, 1 }
    sphere {  m*<-1.8704404628904505,2.2002207141255052,-2.4175993784137555>, 1}
    sphere { m*<-1.6026532418526187,-2.687471228278392,-2.228053093251183>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4755468271701885,0.25903539495150424,7.54589869330394>, <-0.24882510299756053,-0.1282532302930939,-1.443653612997787>, 0.5 }
    cylinder { m*<2.4858832910086965,-0.02621925490671974,-2.6728631384489687>, <-0.24882510299756053,-0.1282532302930939,-1.443653612997787>, 0.5}
    cylinder { m*<-1.8704404628904505,2.2002207141255052,-2.4175993784137555>, <-0.24882510299756053,-0.1282532302930939,-1.443653612997787>, 0.5 }
    cylinder {  m*<-1.6026532418526187,-2.687471228278392,-2.228053093251183>, <-0.24882510299756053,-0.1282532302930939,-1.443653612997787>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    