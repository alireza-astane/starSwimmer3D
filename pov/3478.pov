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
    sphere { m*<0.1631678615685569,0.516323851410271,-0.033512027837610536>, 1 }        
    sphere {  m*<0.4039029663102485,0.6450339295905966,2.9540427432829395>, 1 }
    sphere {  m*<2.897876255574815,0.6183578267966454,-1.2627215532887957>, 1 }
    sphere {  m*<-1.4584474983243338,2.844797795828872,-1.0074577932535815>, 1}
    sphere { m*<-2.8364640180879626,-5.154050159459142,-1.7714798095857112>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4039029663102485,0.6450339295905966,2.9540427432829395>, <0.1631678615685569,0.516323851410271,-0.033512027837610536>, 0.5 }
    cylinder { m*<2.897876255574815,0.6183578267966454,-1.2627215532887957>, <0.1631678615685569,0.516323851410271,-0.033512027837610536>, 0.5}
    cylinder { m*<-1.4584474983243338,2.844797795828872,-1.0074577932535815>, <0.1631678615685569,0.516323851410271,-0.033512027837610536>, 0.5 }
    cylinder {  m*<-2.8364640180879626,-5.154050159459142,-1.7714798095857112>, <0.1631678615685569,0.516323851410271,-0.033512027837610536>, 0.5}

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
    sphere { m*<0.1631678615685569,0.516323851410271,-0.033512027837610536>, 1 }        
    sphere {  m*<0.4039029663102485,0.6450339295905966,2.9540427432829395>, 1 }
    sphere {  m*<2.897876255574815,0.6183578267966454,-1.2627215532887957>, 1 }
    sphere {  m*<-1.4584474983243338,2.844797795828872,-1.0074577932535815>, 1}
    sphere { m*<-2.8364640180879626,-5.154050159459142,-1.7714798095857112>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4039029663102485,0.6450339295905966,2.9540427432829395>, <0.1631678615685569,0.516323851410271,-0.033512027837610536>, 0.5 }
    cylinder { m*<2.897876255574815,0.6183578267966454,-1.2627215532887957>, <0.1631678615685569,0.516323851410271,-0.033512027837610536>, 0.5}
    cylinder { m*<-1.4584474983243338,2.844797795828872,-1.0074577932535815>, <0.1631678615685569,0.516323851410271,-0.033512027837610536>, 0.5 }
    cylinder {  m*<-2.8364640180879626,-5.154050159459142,-1.7714798095857112>, <0.1631678615685569,0.516323851410271,-0.033512027837610536>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    