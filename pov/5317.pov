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
    sphere { m*<-0.654741682025822,-0.15312535370134717,-1.5127095464556353>, 1 }        
    sphere {  m*<0.36929735776043543,0.2872214964213307,8.424955019598258>, 1 }
    sphere {  m*<4.060579827910892,0.01788548443686308,-3.759421523948959>, 1 }
    sphere {  m*<-2.2956385099804755,2.1755873302052438,-2.4532142056526425>, 1}
    sphere { m*<-2.0278512889426437,-2.7121046121986536,-2.2636679204900716>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.36929735776043543,0.2872214964213307,8.424955019598258>, <-0.654741682025822,-0.15312535370134717,-1.5127095464556353>, 0.5 }
    cylinder { m*<4.060579827910892,0.01788548443686308,-3.759421523948959>, <-0.654741682025822,-0.15312535370134717,-1.5127095464556353>, 0.5}
    cylinder { m*<-2.2956385099804755,2.1755873302052438,-2.4532142056526425>, <-0.654741682025822,-0.15312535370134717,-1.5127095464556353>, 0.5 }
    cylinder {  m*<-2.0278512889426437,-2.7121046121986536,-2.2636679204900716>, <-0.654741682025822,-0.15312535370134717,-1.5127095464556353>, 0.5}

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
    sphere { m*<-0.654741682025822,-0.15312535370134717,-1.5127095464556353>, 1 }        
    sphere {  m*<0.36929735776043543,0.2872214964213307,8.424955019598258>, 1 }
    sphere {  m*<4.060579827910892,0.01788548443686308,-3.759421523948959>, 1 }
    sphere {  m*<-2.2956385099804755,2.1755873302052438,-2.4532142056526425>, 1}
    sphere { m*<-2.0278512889426437,-2.7121046121986536,-2.2636679204900716>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.36929735776043543,0.2872214964213307,8.424955019598258>, <-0.654741682025822,-0.15312535370134717,-1.5127095464556353>, 0.5 }
    cylinder { m*<4.060579827910892,0.01788548443686308,-3.759421523948959>, <-0.654741682025822,-0.15312535370134717,-1.5127095464556353>, 0.5}
    cylinder { m*<-2.2956385099804755,2.1755873302052438,-2.4532142056526425>, <-0.654741682025822,-0.15312535370134717,-1.5127095464556353>, 0.5 }
    cylinder {  m*<-2.0278512889426437,-2.7121046121986536,-2.2636679204900716>, <-0.654741682025822,-0.15312535370134717,-1.5127095464556353>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    