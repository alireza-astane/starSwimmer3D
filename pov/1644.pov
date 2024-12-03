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
    sphere { m*<0.8668294834610089,-3.639082247955683e-18,0.8455674388929897>, 1 }        
    sphere {  m*<1.009680001031037,7.084169515918396e-19,3.8421698242871445>, 1 }
    sphere {  m*<5.945135089677579,4.0618845637627994e-18,-1.2271699259552216>, 1 }
    sphere {  m*<-4.000599685465671,8.164965809277259,-2.25952283450102>, 1}
    sphere { m*<-4.000599685465671,-8.164965809277259,-2.2595228345010225>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.009680001031037,7.084169515918396e-19,3.8421698242871445>, <0.8668294834610089,-3.639082247955683e-18,0.8455674388929897>, 0.5 }
    cylinder { m*<5.945135089677579,4.0618845637627994e-18,-1.2271699259552216>, <0.8668294834610089,-3.639082247955683e-18,0.8455674388929897>, 0.5}
    cylinder { m*<-4.000599685465671,8.164965809277259,-2.25952283450102>, <0.8668294834610089,-3.639082247955683e-18,0.8455674388929897>, 0.5 }
    cylinder {  m*<-4.000599685465671,-8.164965809277259,-2.2595228345010225>, <0.8668294834610089,-3.639082247955683e-18,0.8455674388929897>, 0.5}

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
    sphere { m*<0.8668294834610089,-3.639082247955683e-18,0.8455674388929897>, 1 }        
    sphere {  m*<1.009680001031037,7.084169515918396e-19,3.8421698242871445>, 1 }
    sphere {  m*<5.945135089677579,4.0618845637627994e-18,-1.2271699259552216>, 1 }
    sphere {  m*<-4.000599685465671,8.164965809277259,-2.25952283450102>, 1}
    sphere { m*<-4.000599685465671,-8.164965809277259,-2.2595228345010225>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.009680001031037,7.084169515918396e-19,3.8421698242871445>, <0.8668294834610089,-3.639082247955683e-18,0.8455674388929897>, 0.5 }
    cylinder { m*<5.945135089677579,4.0618845637627994e-18,-1.2271699259552216>, <0.8668294834610089,-3.639082247955683e-18,0.8455674388929897>, 0.5}
    cylinder { m*<-4.000599685465671,8.164965809277259,-2.25952283450102>, <0.8668294834610089,-3.639082247955683e-18,0.8455674388929897>, 0.5 }
    cylinder {  m*<-4.000599685465671,-8.164965809277259,-2.2595228345010225>, <0.8668294834610089,-3.639082247955683e-18,0.8455674388929897>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    