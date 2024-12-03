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
    sphere { m*<-1.3627315775635582,-0.17800651337407014,-1.1548029459937403>, 1 }        
    sphere {  m*<0.008746984498311794,0.27968899008918857,8.740085208240492>, 1 }
    sphere {  m*<6.493069007582974,0.09363655842604032,-5.240815131266425>, 1 }
    sphere {  m*<-3.032435982046201,2.1511973008995757,-2.0418685759752018>, 1}
    sphere { m*<-2.7646487610083694,-2.7364946415043216,-1.8523222908126313>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.008746984498311794,0.27968899008918857,8.740085208240492>, <-1.3627315775635582,-0.17800651337407014,-1.1548029459937403>, 0.5 }
    cylinder { m*<6.493069007582974,0.09363655842604032,-5.240815131266425>, <-1.3627315775635582,-0.17800651337407014,-1.1548029459937403>, 0.5}
    cylinder { m*<-3.032435982046201,2.1511973008995757,-2.0418685759752018>, <-1.3627315775635582,-0.17800651337407014,-1.1548029459937403>, 0.5 }
    cylinder {  m*<-2.7646487610083694,-2.7364946415043216,-1.8523222908126313>, <-1.3627315775635582,-0.17800651337407014,-1.1548029459937403>, 0.5}

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
    sphere { m*<-1.3627315775635582,-0.17800651337407014,-1.1548029459937403>, 1 }        
    sphere {  m*<0.008746984498311794,0.27968899008918857,8.740085208240492>, 1 }
    sphere {  m*<6.493069007582974,0.09363655842604032,-5.240815131266425>, 1 }
    sphere {  m*<-3.032435982046201,2.1511973008995757,-2.0418685759752018>, 1}
    sphere { m*<-2.7646487610083694,-2.7364946415043216,-1.8523222908126313>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.008746984498311794,0.27968899008918857,8.740085208240492>, <-1.3627315775635582,-0.17800651337407014,-1.1548029459937403>, 0.5 }
    cylinder { m*<6.493069007582974,0.09363655842604032,-5.240815131266425>, <-1.3627315775635582,-0.17800651337407014,-1.1548029459937403>, 0.5}
    cylinder { m*<-3.032435982046201,2.1511973008995757,-2.0418685759752018>, <-1.3627315775635582,-0.17800651337407014,-1.1548029459937403>, 0.5 }
    cylinder {  m*<-2.7646487610083694,-2.7364946415043216,-1.8523222908126313>, <-1.3627315775635582,-0.17800651337407014,-1.1548029459937403>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    