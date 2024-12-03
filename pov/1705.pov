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
    sphere { m*<0.9425735319425611,-1.9261871940804533e-18,0.8083163007636894>, 1 }        
    sphere {  m*<1.1020416733620875,9.979935313389344e-19,3.8040810741085025>, 1 }
    sphere {  m*<5.614856600781135,5.613425431171004e-18,-1.1290406475279415>, 1 }
    sphere {  m*<-3.942003483127801,8.164965809277259,-2.2697312158839695>, 1}
    sphere { m*<-3.942003483127801,-8.164965809277259,-2.269731215883973>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1020416733620875,9.979935313389344e-19,3.8040810741085025>, <0.9425735319425611,-1.9261871940804533e-18,0.8083163007636894>, 0.5 }
    cylinder { m*<5.614856600781135,5.613425431171004e-18,-1.1290406475279415>, <0.9425735319425611,-1.9261871940804533e-18,0.8083163007636894>, 0.5}
    cylinder { m*<-3.942003483127801,8.164965809277259,-2.2697312158839695>, <0.9425735319425611,-1.9261871940804533e-18,0.8083163007636894>, 0.5 }
    cylinder {  m*<-3.942003483127801,-8.164965809277259,-2.269731215883973>, <0.9425735319425611,-1.9261871940804533e-18,0.8083163007636894>, 0.5}

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
    sphere { m*<0.9425735319425611,-1.9261871940804533e-18,0.8083163007636894>, 1 }        
    sphere {  m*<1.1020416733620875,9.979935313389344e-19,3.8040810741085025>, 1 }
    sphere {  m*<5.614856600781135,5.613425431171004e-18,-1.1290406475279415>, 1 }
    sphere {  m*<-3.942003483127801,8.164965809277259,-2.2697312158839695>, 1}
    sphere { m*<-3.942003483127801,-8.164965809277259,-2.269731215883973>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1020416733620875,9.979935313389344e-19,3.8040810741085025>, <0.9425735319425611,-1.9261871940804533e-18,0.8083163007636894>, 0.5 }
    cylinder { m*<5.614856600781135,5.613425431171004e-18,-1.1290406475279415>, <0.9425735319425611,-1.9261871940804533e-18,0.8083163007636894>, 0.5}
    cylinder { m*<-3.942003483127801,8.164965809277259,-2.2697312158839695>, <0.9425735319425611,-1.9261871940804533e-18,0.8083163007636894>, 0.5 }
    cylinder {  m*<-3.942003483127801,-8.164965809277259,-2.269731215883973>, <0.9425735319425611,-1.9261871940804533e-18,0.8083163007636894>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    