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
    sphere { m*<-1.3622534557982373,-0.5572837468132179,-0.9078743322119195>, 1 }        
    sphere {  m*<0.09055556231987172,0.06043321507637059,8.966810135049421>, 1 }
    sphere {  m*<7.445907000319843,-0.028487060917986556,-5.612683154995933>, 1 }
    sphere {  m*<-4.513574816683488,3.5210395939929278,-2.5227068226533276>, 1}
    sphere { m*<-2.690018989068364,-3.166444037642631,-1.5618164310830374>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.09055556231987172,0.06043321507637059,8.966810135049421>, <-1.3622534557982373,-0.5572837468132179,-0.9078743322119195>, 0.5 }
    cylinder { m*<7.445907000319843,-0.028487060917986556,-5.612683154995933>, <-1.3622534557982373,-0.5572837468132179,-0.9078743322119195>, 0.5}
    cylinder { m*<-4.513574816683488,3.5210395939929278,-2.5227068226533276>, <-1.3622534557982373,-0.5572837468132179,-0.9078743322119195>, 0.5 }
    cylinder {  m*<-2.690018989068364,-3.166444037642631,-1.5618164310830374>, <-1.3622534557982373,-0.5572837468132179,-0.9078743322119195>, 0.5}

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
    sphere { m*<-1.3622534557982373,-0.5572837468132179,-0.9078743322119195>, 1 }        
    sphere {  m*<0.09055556231987172,0.06043321507637059,8.966810135049421>, 1 }
    sphere {  m*<7.445907000319843,-0.028487060917986556,-5.612683154995933>, 1 }
    sphere {  m*<-4.513574816683488,3.5210395939929278,-2.5227068226533276>, 1}
    sphere { m*<-2.690018989068364,-3.166444037642631,-1.5618164310830374>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.09055556231987172,0.06043321507637059,8.966810135049421>, <-1.3622534557982373,-0.5572837468132179,-0.9078743322119195>, 0.5 }
    cylinder { m*<7.445907000319843,-0.028487060917986556,-5.612683154995933>, <-1.3622534557982373,-0.5572837468132179,-0.9078743322119195>, 0.5}
    cylinder { m*<-4.513574816683488,3.5210395939929278,-2.5227068226533276>, <-1.3622534557982373,-0.5572837468132179,-0.9078743322119195>, 0.5 }
    cylinder {  m*<-2.690018989068364,-3.166444037642631,-1.5618164310830374>, <-1.3622534557982373,-0.5572837468132179,-0.9078743322119195>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    