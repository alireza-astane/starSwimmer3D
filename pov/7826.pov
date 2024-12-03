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
    sphere { m*<-0.4062253118603275,-0.40521964288019696,-0.43768189004001845>, 1 }        
    sphere {  m*<1.0129421823398337,0.5847192709997202,9.411608206995126>, 1 }
    sphere {  m*<8.380729380662629,0.29962702020745846,-5.159069222078802>, 1 }
    sphere {  m*<-6.515233813026363,6.822708393828093,-3.6682623188971952>, 1}
    sphere { m*<-4.007178530096464,-8.247392637939937,-2.1052384912727544>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0129421823398337,0.5847192709997202,9.411608206995126>, <-0.4062253118603275,-0.40521964288019696,-0.43768189004001845>, 0.5 }
    cylinder { m*<8.380729380662629,0.29962702020745846,-5.159069222078802>, <-0.4062253118603275,-0.40521964288019696,-0.43768189004001845>, 0.5}
    cylinder { m*<-6.515233813026363,6.822708393828093,-3.6682623188971952>, <-0.4062253118603275,-0.40521964288019696,-0.43768189004001845>, 0.5 }
    cylinder {  m*<-4.007178530096464,-8.247392637939937,-2.1052384912727544>, <-0.4062253118603275,-0.40521964288019696,-0.43768189004001845>, 0.5}

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
    sphere { m*<-0.4062253118603275,-0.40521964288019696,-0.43768189004001845>, 1 }        
    sphere {  m*<1.0129421823398337,0.5847192709997202,9.411608206995126>, 1 }
    sphere {  m*<8.380729380662629,0.29962702020745846,-5.159069222078802>, 1 }
    sphere {  m*<-6.515233813026363,6.822708393828093,-3.6682623188971952>, 1}
    sphere { m*<-4.007178530096464,-8.247392637939937,-2.1052384912727544>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0129421823398337,0.5847192709997202,9.411608206995126>, <-0.4062253118603275,-0.40521964288019696,-0.43768189004001845>, 0.5 }
    cylinder { m*<8.380729380662629,0.29962702020745846,-5.159069222078802>, <-0.4062253118603275,-0.40521964288019696,-0.43768189004001845>, 0.5}
    cylinder { m*<-6.515233813026363,6.822708393828093,-3.6682623188971952>, <-0.4062253118603275,-0.40521964288019696,-0.43768189004001845>, 0.5 }
    cylinder {  m*<-4.007178530096464,-8.247392637939937,-2.1052384912727544>, <-0.4062253118603275,-0.40521964288019696,-0.43768189004001845>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    