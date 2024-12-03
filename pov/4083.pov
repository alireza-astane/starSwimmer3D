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
    sphere { m*<-0.15732150326909053,-0.07933042990853958,-0.30808173104553327>, 1 }        
    sphere {  m*<0.13059768200197247,0.07460682359512932,3.2650337752146443>, 1 }
    sphere {  m*<2.5773868907371664,0.022703545477834527,-1.537291256496717>, 1 }
    sphere {  m*<-1.7789368631619806,2.2491435145100596,-1.2820274964615035>, 1}
    sphere { m*<-1.5111496421241488,-2.638548427893838,-1.0924812112989308>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.13059768200197247,0.07460682359512932,3.2650337752146443>, <-0.15732150326909053,-0.07933042990853958,-0.30808173104553327>, 0.5 }
    cylinder { m*<2.5773868907371664,0.022703545477834527,-1.537291256496717>, <-0.15732150326909053,-0.07933042990853958,-0.30808173104553327>, 0.5}
    cylinder { m*<-1.7789368631619806,2.2491435145100596,-1.2820274964615035>, <-0.15732150326909053,-0.07933042990853958,-0.30808173104553327>, 0.5 }
    cylinder {  m*<-1.5111496421241488,-2.638548427893838,-1.0924812112989308>, <-0.15732150326909053,-0.07933042990853958,-0.30808173104553327>, 0.5}

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
    sphere { m*<-0.15732150326909053,-0.07933042990853958,-0.30808173104553327>, 1 }        
    sphere {  m*<0.13059768200197247,0.07460682359512932,3.2650337752146443>, 1 }
    sphere {  m*<2.5773868907371664,0.022703545477834527,-1.537291256496717>, 1 }
    sphere {  m*<-1.7789368631619806,2.2491435145100596,-1.2820274964615035>, 1}
    sphere { m*<-1.5111496421241488,-2.638548427893838,-1.0924812112989308>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.13059768200197247,0.07460682359512932,3.2650337752146443>, <-0.15732150326909053,-0.07933042990853958,-0.30808173104553327>, 0.5 }
    cylinder { m*<2.5773868907371664,0.022703545477834527,-1.537291256496717>, <-0.15732150326909053,-0.07933042990853958,-0.30808173104553327>, 0.5}
    cylinder { m*<-1.7789368631619806,2.2491435145100596,-1.2820274964615035>, <-0.15732150326909053,-0.07933042990853958,-0.30808173104553327>, 0.5 }
    cylinder {  m*<-1.5111496421241488,-2.638548427893838,-1.0924812112989308>, <-0.15732150326909053,-0.07933042990853958,-0.30808173104553327>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    