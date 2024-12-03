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
    sphere { m*<-0.6694555810611471,-0.9784837714322843,-0.5595805503245057>, 1 }        
    sphere {  m*<0.7497119131390154,0.01145514244763346,9.289709546710647>, 1 }
    sphere {  m*<8.117499111461811,-0.2736371083446293,-5.280967882363287>, 1 }
    sphere {  m*<-6.778464082227175,6.249444265276017,-3.7901609791816817>, 1}
    sphere { m*<-2.7618515402751616,-5.535313613936561,-1.5285430673907912>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7497119131390154,0.01145514244763346,9.289709546710647>, <-0.6694555810611471,-0.9784837714322843,-0.5595805503245057>, 0.5 }
    cylinder { m*<8.117499111461811,-0.2736371083446293,-5.280967882363287>, <-0.6694555810611471,-0.9784837714322843,-0.5595805503245057>, 0.5}
    cylinder { m*<-6.778464082227175,6.249444265276017,-3.7901609791816817>, <-0.6694555810611471,-0.9784837714322843,-0.5595805503245057>, 0.5 }
    cylinder {  m*<-2.7618515402751616,-5.535313613936561,-1.5285430673907912>, <-0.6694555810611471,-0.9784837714322843,-0.5595805503245057>, 0.5}

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
    sphere { m*<-0.6694555810611471,-0.9784837714322843,-0.5595805503245057>, 1 }        
    sphere {  m*<0.7497119131390154,0.01145514244763346,9.289709546710647>, 1 }
    sphere {  m*<8.117499111461811,-0.2736371083446293,-5.280967882363287>, 1 }
    sphere {  m*<-6.778464082227175,6.249444265276017,-3.7901609791816817>, 1}
    sphere { m*<-2.7618515402751616,-5.535313613936561,-1.5285430673907912>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7497119131390154,0.01145514244763346,9.289709546710647>, <-0.6694555810611471,-0.9784837714322843,-0.5595805503245057>, 0.5 }
    cylinder { m*<8.117499111461811,-0.2736371083446293,-5.280967882363287>, <-0.6694555810611471,-0.9784837714322843,-0.5595805503245057>, 0.5}
    cylinder { m*<-6.778464082227175,6.249444265276017,-3.7901609791816817>, <-0.6694555810611471,-0.9784837714322843,-0.5595805503245057>, 0.5 }
    cylinder {  m*<-2.7618515402751616,-5.535313613936561,-1.5285430673907912>, <-0.6694555810611471,-0.9784837714322843,-0.5595805503245057>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    