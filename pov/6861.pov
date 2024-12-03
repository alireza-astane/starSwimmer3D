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
    sphere { m*<-0.9307000337787622,-1.1254015309704815,-0.6866745198525779>, 1 }        
    sphere {  m*<0.49795179974096393,-0.27730515297750724,9.174414286171425>, 1 }
    sphere {  m*<7.8533032377409375,-0.3662254289718634,-5.4050790038739045>, 1 }
    sphere {  m*<-6.442598527174918,5.453003975521616,-3.507747313095528>, 1}
    sphere { m*<-2.163919304651352,-3.7919374343023744,-1.2925653566590343>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.49795179974096393,-0.27730515297750724,9.174414286171425>, <-0.9307000337787622,-1.1254015309704815,-0.6866745198525779>, 0.5 }
    cylinder { m*<7.8533032377409375,-0.3662254289718634,-5.4050790038739045>, <-0.9307000337787622,-1.1254015309704815,-0.6866745198525779>, 0.5}
    cylinder { m*<-6.442598527174918,5.453003975521616,-3.507747313095528>, <-0.9307000337787622,-1.1254015309704815,-0.6866745198525779>, 0.5 }
    cylinder {  m*<-2.163919304651352,-3.7919374343023744,-1.2925653566590343>, <-0.9307000337787622,-1.1254015309704815,-0.6866745198525779>, 0.5}

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
    sphere { m*<-0.9307000337787622,-1.1254015309704815,-0.6866745198525779>, 1 }        
    sphere {  m*<0.49795179974096393,-0.27730515297750724,9.174414286171425>, 1 }
    sphere {  m*<7.8533032377409375,-0.3662254289718634,-5.4050790038739045>, 1 }
    sphere {  m*<-6.442598527174918,5.453003975521616,-3.507747313095528>, 1}
    sphere { m*<-2.163919304651352,-3.7919374343023744,-1.2925653566590343>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.49795179974096393,-0.27730515297750724,9.174414286171425>, <-0.9307000337787622,-1.1254015309704815,-0.6866745198525779>, 0.5 }
    cylinder { m*<7.8533032377409375,-0.3662254289718634,-5.4050790038739045>, <-0.9307000337787622,-1.1254015309704815,-0.6866745198525779>, 0.5}
    cylinder { m*<-6.442598527174918,5.453003975521616,-3.507747313095528>, <-0.9307000337787622,-1.1254015309704815,-0.6866745198525779>, 0.5 }
    cylinder {  m*<-2.163919304651352,-3.7919374343023744,-1.2925653566590343>, <-0.9307000337787622,-1.1254015309704815,-0.6866745198525779>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    