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
    sphere { m*<0.004826813256660567,0.21700280095377122,-0.1252538320105301>, 1 }        
    sphere {  m*<0.2455619179983523,0.34571287913409665,2.8623009391100216>, 1 }
    sphere {  m*<2.739535207262923,0.3190367763401457,-1.3544633574617166>, 1 }
    sphere {  m*<-1.616788546636231,2.5454767453723735,-1.0991995974265016>, 1}
    sphere { m*<-2.2303434439930307,-4.008266113719943,-1.420297373908881>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2455619179983523,0.34571287913409665,2.8623009391100216>, <0.004826813256660567,0.21700280095377122,-0.1252538320105301>, 0.5 }
    cylinder { m*<2.739535207262923,0.3190367763401457,-1.3544633574617166>, <0.004826813256660567,0.21700280095377122,-0.1252538320105301>, 0.5}
    cylinder { m*<-1.616788546636231,2.5454767453723735,-1.0991995974265016>, <0.004826813256660567,0.21700280095377122,-0.1252538320105301>, 0.5 }
    cylinder {  m*<-2.2303434439930307,-4.008266113719943,-1.420297373908881>, <0.004826813256660567,0.21700280095377122,-0.1252538320105301>, 0.5}

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
    sphere { m*<0.004826813256660567,0.21700280095377122,-0.1252538320105301>, 1 }        
    sphere {  m*<0.2455619179983523,0.34571287913409665,2.8623009391100216>, 1 }
    sphere {  m*<2.739535207262923,0.3190367763401457,-1.3544633574617166>, 1 }
    sphere {  m*<-1.616788546636231,2.5454767453723735,-1.0991995974265016>, 1}
    sphere { m*<-2.2303434439930307,-4.008266113719943,-1.420297373908881>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2455619179983523,0.34571287913409665,2.8623009391100216>, <0.004826813256660567,0.21700280095377122,-0.1252538320105301>, 0.5 }
    cylinder { m*<2.739535207262923,0.3190367763401457,-1.3544633574617166>, <0.004826813256660567,0.21700280095377122,-0.1252538320105301>, 0.5}
    cylinder { m*<-1.616788546636231,2.5454767453723735,-1.0991995974265016>, <0.004826813256660567,0.21700280095377122,-0.1252538320105301>, 0.5 }
    cylinder {  m*<-2.2303434439930307,-4.008266113719943,-1.420297373908881>, <0.004826813256660567,0.21700280095377122,-0.1252538320105301>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    