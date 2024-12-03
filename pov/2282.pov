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
    sphere { m*<1.0602123674193302,0.3744213035064217,0.49273538154669105>, 1 }        
    sphere {  m*<1.3042427078366086,0.40399266879475815,3.4826462858910965>, 1 }
    sphere {  m*<3.797489896899143,0.40399266879475804,-0.7346359225995209>, 1 }
    sphere {  m*<-3.020778705996545,6.841842275031472,-1.9202108556530069>, 1}
    sphere { m*<-3.788421828232092,-7.88792059193573,-2.373415499361454>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3042427078366086,0.40399266879475815,3.4826462858910965>, <1.0602123674193302,0.3744213035064217,0.49273538154669105>, 0.5 }
    cylinder { m*<3.797489896899143,0.40399266879475804,-0.7346359225995209>, <1.0602123674193302,0.3744213035064217,0.49273538154669105>, 0.5}
    cylinder { m*<-3.020778705996545,6.841842275031472,-1.9202108556530069>, <1.0602123674193302,0.3744213035064217,0.49273538154669105>, 0.5 }
    cylinder {  m*<-3.788421828232092,-7.88792059193573,-2.373415499361454>, <1.0602123674193302,0.3744213035064217,0.49273538154669105>, 0.5}

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
    sphere { m*<1.0602123674193302,0.3744213035064217,0.49273538154669105>, 1 }        
    sphere {  m*<1.3042427078366086,0.40399266879475815,3.4826462858910965>, 1 }
    sphere {  m*<3.797489896899143,0.40399266879475804,-0.7346359225995209>, 1 }
    sphere {  m*<-3.020778705996545,6.841842275031472,-1.9202108556530069>, 1}
    sphere { m*<-3.788421828232092,-7.88792059193573,-2.373415499361454>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3042427078366086,0.40399266879475815,3.4826462858910965>, <1.0602123674193302,0.3744213035064217,0.49273538154669105>, 0.5 }
    cylinder { m*<3.797489896899143,0.40399266879475804,-0.7346359225995209>, <1.0602123674193302,0.3744213035064217,0.49273538154669105>, 0.5}
    cylinder { m*<-3.020778705996545,6.841842275031472,-1.9202108556530069>, <1.0602123674193302,0.3744213035064217,0.49273538154669105>, 0.5 }
    cylinder {  m*<-3.788421828232092,-7.88792059193573,-2.373415499361454>, <1.0602123674193302,0.3744213035064217,0.49273538154669105>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    